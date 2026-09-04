const statusEl = document.getElementById("status");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadMessage = document.getElementById("upload-message");
const documentList = document.getElementById("document-list");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const answerEl = document.getElementById("answer");
const citationsEl = document.getElementById("citations");

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

function setMessage(text, isError = false) {
  uploadMessage.textContent = text;
  uploadMessage.classList.toggle("error", isError);
}

function renderDocuments(documents) {
  documentList.innerHTML = "";
  if (documents.length === 0) {
    const empty = document.createElement("li");
    empty.textContent = "No documents indexed yet.";
    documentList.append(empty);
    return;
  }
  for (const doc of documents) {
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${doc.name} · ${doc.chunks} chunks`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "✕";
    remove.title = "Remove document";
    remove.addEventListener("click", async () => {
      try {
        await request(`/api/documents/${doc.id}`, { method: "DELETE" });
        await refreshStatus();
      } catch (error) {
        setMessage(error.message, true);
      }
    });
    item.append(label, remove);
    documentList.append(item);
  }
}

async function refreshStatus() {
  const status = await request("/api/status");
  statusEl.textContent = `Embeddings: ${status.embedding_backend} · LLM: ${
    status.llm_enabled ? "enabled" : "not configured (set OPENAI_API_KEY)"
  }`;
  renderDocuments(status.documents);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  const button = uploadForm.querySelector("button");
  button.disabled = true;
  setMessage(`Indexing ${file.name}…`);
  try {
    const body = new FormData();
    body.append("file", file);
    const result = await request("/api/documents", { method: "POST", body });
    setMessage(`Indexed ${result.name} into ${result.chunks} chunks.`);
    uploadForm.reset();
    await refreshStatus();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    button.disabled = false;
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  const button = askForm.querySelector("button");
  button.disabled = true;
  answerEl.classList.remove("hidden");
  answerEl.textContent = "Thinking…";
  citationsEl.innerHTML = "";
  try {
    const result = await request("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    answerEl.textContent = result.answer;
    for (const [index, citation] of result.citations.entries()) {
      const box = document.createElement("div");
      box.className = "citation";
      const title = document.createElement("strong");
      title.textContent = `[${index + 1}] ${citation.document_name} · chunk ${citation.position} · score ${citation.score}`;
      const excerpt = document.createElement("p");
      excerpt.textContent = citation.excerpt;
      box.append(title, excerpt);
      citationsEl.append(box);
    }
  } catch (error) {
    answerEl.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

refreshStatus().catch((error) => {
  statusEl.textContent = error.message;
});
