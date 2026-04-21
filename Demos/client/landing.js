const bipedLink = document.getElementById("biped-link");
const quadrupedLink = document.getElementById("quadruped-link");
const queueStatus = document.getElementById("queue-status");
const queueMessage = document.getElementById("queue-message");
const queueCountdown = document.getElementById("queue-countdown");

const POLL_INTERVAL = 3000;
const FETCH_OPTS = { credentials: "same-origin" };

let sessionExpiresAt = null;
let pollTimer = null;
let countdownTimer = null;
let sid = sessionStorage.getItem("q_sid") || null;

function saveSid(nextSid) {
    sid = nextSid || null;
    if (sid) sessionStorage.setItem("q_sid", sid);
    else sessionStorage.removeItem("q_sid");
}

function formatTime(s) {
    if (s < 60) return `${Math.ceil(s)}s`;
    return `${Math.floor(s / 60)}m ${Math.floor(s % 60).toString().padStart(2, "0")}s`;
}

function retryDelayMs(res, data, fallbackMs) {
    if (data && typeof data.retry_after_seconds === "number" && Number.isFinite(data.retry_after_seconds)) {
        return Math.max(1000, data.retry_after_seconds * 1000);
    }
    const h = res.headers.get("Retry-After");
    if (h) {
        const s = parseInt(h, 10);
        if (Number.isFinite(s) && s >= 0) return Math.max(1000, s * 1000);
    }
    return fallbackMs;
}

function setCardsWaiting(waiting) {
    bipedLink.classList.toggle("waiting", waiting);
    quadrupedLink.classList.toggle("waiting", waiting);
}

function updateDemoLinks() {
    const sidParam = sid ? `?sid=${encodeURIComponent(sid)}` : "";
    bipedLink.href = `/biped.html${sidParam}`;
    quadrupedLink.href = `/quadruped.html${sidParam}`;
}

function showPromoted(remainingSeconds) {
    setCardsWaiting(false);
    updateDemoLinks();
    queueStatus.classList.remove("hidden", "queue-waiting");
    queueStatus.classList.add("queue-your-turn");
    queueMessage.textContent = "Select a demo below.";
    queueCountdown.classList.remove("hidden");
    startTokenCountdown(remainingSeconds);
}

function showPosition(position, estimatedWaitSeconds, queueLength) {
    setCardsWaiting(true);
    updateDemoLinks();
    queueStatus.classList.remove("hidden", "queue-your-turn");
    queueStatus.classList.add("queue-waiting");
    let line;
    if (typeof queueLength === "number" && queueLength >= position) {
        line = `You are #${position} of ${queueLength} in queue. Estimated wait: ~${formatTime(estimatedWaitSeconds)}.`;
    } else {
        line = `You are #${position} in queue. Estimated wait: ~${formatTime(estimatedWaitSeconds)}.`;
    }
    queueMessage.textContent = line;
    queueCountdown.classList.add("hidden");
    stopTokenCountdown();
}

function showConnecting() {
    setCardsWaiting(true);
    queueStatus.classList.remove("hidden", "queue-your-turn", "queue-waiting");
    queueMessage.textContent = "Connecting...";
    queueCountdown.classList.add("hidden");
    stopTokenCountdown();
}

function startTokenCountdown(remainingSeconds) {
    stopTokenCountdown();
    sessionExpiresAt = Date.now() + remainingSeconds * 1000;
    updateCountdownDisplay();
    countdownTimer = setInterval(updateCountdownDisplay, 1000);
}

function stopTokenCountdown() {
    if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
    }
}

function updateCountdownDisplay() {
    if (!sessionExpiresAt) return;
    const remaining = Math.max(0, Math.ceil((sessionExpiresAt - Date.now()) / 1000));
    queueCountdown.textContent = `Session expires in ${formatTime(remaining)}`;
    if (remaining <= 0) {
        stopTokenCountdown();
        queueCountdown.classList.add("hidden");
        stopPolling();
        joinQueue();
    }
}

function handleResponse(data) {
    if (typeof data.sid === "string" && data.sid) saveSid(data.sid);
    const queueLength = typeof data.queueLength === "number" ? data.queueLength : null;
    if (data.promoted) {
        showPromoted(data.remainingSeconds);
    } else {
        showPosition(data.position, data.estimatedWaitSeconds, queueLength);
    }
}

function showError(msg) {
    setCardsWaiting(true);
    queueStatus.classList.remove("hidden", "queue-your-turn", "queue-waiting");
    queueStatus.classList.add("queue-waiting");
    queueMessage.textContent = msg;
    queueCountdown.classList.add("hidden");
    stopTokenCountdown();
}

async function joinQueue() {
    stopPolling();
    showConnecting();
    try {
        const res = await fetch("/api/queue/join", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sid }),
            ...FETCH_OPTS,
        });
        const data = await res.json().catch(() => ({}));
        if (res.status === 429) {
            const delay = retryDelayMs(res, data, 10000);
            showError("Reconnecting...");
            setTimeout(joinQueue, delay);
            return;
        }
        if (res.status === 503) {
            const delay = retryDelayMs(res, data, 15000);
            showError(data.message || "Queue is full. Please try again later.");
            setTimeout(joinQueue, delay);
            return;
        }
        if (!res.ok) {
            showError("Something went wrong. Retrying...");
            setTimeout(joinQueue, 5000);
            return;
        }
        handleResponse(data);
        startPolling();
    } catch {
        showError("Failed to connect. Retrying...");
        setTimeout(joinQueue, 3000);
    }
}

async function pollStatus() {
    try {
        if (!sid) {
            setTimeout(joinQueue, 0);
            return;
        }
        const res = await fetch(`/api/queue/status?sid=${encodeURIComponent(sid)}`, FETCH_OPTS);
        if (res.status === 429) return;
        if (res.status === 400 || res.status === 404) {
            let msg = "Reconnecting to the queue...";
            try {
                const err = await res.json();
                if (err.message) msg = err.message;
            } catch {
                /* ignore */
            }
            saveSid(null);
            stopPolling();
            showError(msg);
            setTimeout(joinQueue, 2500);
            return;
        }
        if (!res.ok) return;
        const data = await res.json();
        handleResponse(data);
    } catch {
        /* network error, keep polling */
    }
}

function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollStatus, POLL_INTERVAL);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

async function init() {
    try {
        sessionStorage.removeItem("q_ticket");
        sessionStorage.removeItem("q_token");
    } catch {
        /* ignore */
    }
    await joinQueue();
}

init();
