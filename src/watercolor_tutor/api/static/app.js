// Watercolor Tutor — frontend logic.
//
// A tiny vanilla-JS single-page app. It talks to the FastAPI backend that serves
// it (same origin, so no CORS) and keeps just one piece of state: the current
// thread_id. Everything else — message history, step, status — comes back fresh
// from the API on each call, mirroring the stateless-per-request backend design.

const $ = (id) => document.getElementById(id);

let currentThreadId = null;

// --- API helpers ------------------------------------------------------------

// Thin fetch wrapper: returns parsed JSON, throws a readable error on failure.
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_) {
      /* non-JSON error body — keep statusText */
    }
    throw new Error(detail);
  }
  return response.json();
}

// --- View switching ---------------------------------------------------------

function showStart() {
  $("chat-view").hidden = true;
  $("start-view").hidden = false;
  loadSessions();
}

function showChat() {
  $("start-view").hidden = true;
  $("chat-view").hidden = false;
  $("text-input").focus();
}

// --- Rendering --------------------------------------------------------------

function scrollToBottom() {
  const box = $("messages");
  box.scrollTop = box.scrollHeight;
}

// Append a chat bubble. Tutor messages are markdown -> sanitized HTML; the
// learner's own text is rendered as plain text (never interpreted as markup).
function addBubble(role, content, { markdown = false, extraClass = "" } = {}) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}${extraClass ? " " + extraClass : ""}`;
  if (markdown) {
    bubble.innerHTML = DOMPurify.sanitize(marked.parse(content));
  } else {
    bubble.textContent = content;
  }
  $("messages").appendChild(bubble);
  scrollToBottom();
  return bubble;
}

// A learner bubble showing the uploaded painting (thumbnail) plus optional caption.
function addImageBubble(file, caption) {
  const bubble = document.createElement("div");
  bubble.className = "bubble learner";
  const img = document.createElement("img");
  img.className = "painting";
  img.src = URL.createObjectURL(file); // shown locally; no server round-trip needed
  bubble.appendChild(img);
  if (caption) {
    const p = document.createElement("div");
    p.textContent = caption;
    bubble.appendChild(p);
  }
  $("messages").appendChild(bubble);
  scrollToBottom();
}

// A transient "tutor is thinking…" bubble; returns a remover.
function showTyping() {
  const bubble = addBubble("tutor", "…", { extraClass: "typing" });
  return () => bubble.remove();
}

function renderHistory(messages) {
  $("messages").innerHTML = "";
  for (const m of messages) {
    const isTutor = m.role === "assistant";
    addBubble(isTutor ? "tutor" : "learner", m.content, { markdown: isTutor });
  }
}

// Update the header (name + progress) and lock input once the lesson is complete.
function applyStatus(resp) {
  $("learner-name").textContent = resp.name || resp.thread_id;
  const done = resp.status === "complete";
  $("progress").textContent = done
    ? "Lesson complete 🎉"
    : `Step ${resp.step} of ${resp.total_steps}`;
  for (const id of ["text-input", "send-button", "upload-button", "file-input"]) {
    $(id).toggleAttribute("disabled", done);
  }
  $("text-input").placeholder = done ? "Lesson complete" : "Type a message…";
}

// --- Session flows ----------------------------------------------------------

async function loadSessions() {
  try {
    const sessions = await api("/sessions");
    const list = $("sessions-list");
    list.innerHTML = "";
    $("sessions-section").hidden = sessions.length === 0;
    for (const s of sessions) {
      const li = document.createElement("li");
      li.className = "session-row";
      const meta =
        s.status === "complete" ? "✓ complete" : `Step ${s.step}`;
      li.innerHTML = `<span>${s.name}</span><span class="meta">${meta}</span>`;
      li.addEventListener("click", () => resumeSession(s.thread_id));
      list.appendChild(li);
    }
  } catch (err) {
    // A missing list is non-fatal — the learner can still start a new session.
    console.warn("Could not load sessions:", err.message);
  }
}

// Enter a session (resume if the name exists, else start fresh) — one POST.
async function startSession(name) {
  const resp = await api("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  enterSession(resp);
}

async function resumeSession(threadId) {
  enterSession(await api(`/sessions/${encodeURIComponent(threadId)}`));
}

function enterSession(resp) {
  currentThreadId = resp.thread_id;
  renderHistory(resp.messages);
  applyStatus(resp);
  showChat();
}

// --- Turns ------------------------------------------------------------------

async function sendMessage(text) {
  addBubble("learner", text, { markdown: false });
  const stopTyping = showTyping();
  try {
    const resp = await api(`/sessions/${encodeURIComponent(currentThreadId)}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    stopTyping();
    resp.messages.forEach((m) => addBubble("tutor", m.content, { markdown: true }));
    applyStatus(resp);
  } catch (err) {
    stopTyping();
    addBubble("tutor", `⚠️ ${err.message}`, { extraClass: "typing" });
  }
}

async function uploadPainting(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const caption = $("text-input").value.trim();
  $("text-input").value = "";
  addImageBubble(file, caption);
  const stopTyping = showTyping();
  try {
    const form = new FormData();
    form.append("file", file);
    if (caption) form.append("text", caption);
    const resp = await api(`/sessions/${encodeURIComponent(currentThreadId)}/feedback`, {
      method: "POST",
      body: form, // multipart; let the browser set the boundary header
    });
    stopTyping();
    resp.messages.forEach((m) => addBubble("tutor", m.content, { markdown: true }));
    applyStatus(resp);
  } catch (err) {
    stopTyping();
    addBubble("tutor", `⚠️ ${err.message}`, { extraClass: "typing" });
  }
}

// --- Wiring -----------------------------------------------------------------

$("start-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = $("name-input").value.trim();
  if (name) startSession(name);
});

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("text-input").value.trim();
  if (!text) return;
  $("text-input").value = "";
  sendMessage(text);
});

$("file-input").addEventListener("change", (e) => {
  if (e.target.files.length) uploadPainting(e.target.files[0]);
  e.target.value = ""; // allow re-uploading the same file
});

$("back-button").addEventListener("click", showStart);

// Drag-and-drop a painting anywhere on the chat.
const chatView = $("chat-view");
["dragenter", "dragover"].forEach((evt) =>
  chatView.addEventListener(evt, (e) => {
    e.preventDefault();
    chatView.classList.add("dragging");
  })
);
["dragleave", "drop"].forEach((evt) =>
  chatView.addEventListener(evt, (e) => {
    e.preventDefault();
    chatView.classList.remove("dragging");
  })
);
chatView.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) uploadPainting(e.dataTransfer.files[0]);
});

// Start on the session screen.
showStart();
