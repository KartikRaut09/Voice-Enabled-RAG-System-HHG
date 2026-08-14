/**
 * HH GOA 2026 — Voice RAG Application
 * Handles text query, voice recording, and response rendering.
 * API endpoints: GET /health, POST /api/query, POST /api/voice-query
 */
(function () {
    'use strict';

    const API_BASE = window.location.origin;

    // ── DOM References ──
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const micBtn = document.getElementById('mic-btn');
    const micIcon = document.getElementById('mic-icon');
    const stopIcon = document.getElementById('stop-icon');
    const voiceState = document.getElementById('voice-state');
    const voicePulse = document.getElementById('voice-pulse');
    const voiceStateText = document.getElementById('voice-state-text');
    const resultsSection = document.getElementById('results-section');
    const transcriptionCard = document.getElementById('transcription-card');
    const transcriptionText = document.getElementById('transcription-text');
    const answerText = document.getElementById('answer-text');
    const answerStatus = document.getElementById('answer-status');
    const sourcesSection = document.getElementById('sources-section');
    const sourcesGrid = document.getElementById('sources-grid');
    const latencyBody = document.getElementById('latency-body');
    const loadingSection = document.getElementById('loading-section');
    const loadingText = document.getElementById('loading-text');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');
    const errorDismiss = document.getElementById('error-dismiss');

    // ── Voice Recording State ──
    let mediaRecorder = null;
    let audioChunks = [];
    let voiceMode = 'IDLE'; // IDLE | RECORDING | TRANSCRIBING | PROCESSING | SUCCESS | ERROR

    // ── Health Check ──
    async function checkHealth() {
        try {
            statusIndicator.className = 'status-badge checking';
            statusText.textContent = 'Checking…';
            const res = await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(5000) });
            if (res.ok) {
                const data = await res.json();
                statusIndicator.className = 'status-badge connected';
                statusText.textContent = data.app_name + ' v' + data.version;
            } else {
                throw new Error('Health check failed');
            }
        } catch {
            statusIndicator.className = 'status-badge disconnected';
            statusText.textContent = 'Disconnected';
        }
    }

    // ── UI State Management ──
    function showLoading(message) {
        loadingText.textContent = message || 'Processing query…';
        loadingSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
    }

    function hideLoading() {
        loadingSection.classList.add('hidden');
    }

    function showError(msg) {
        hideLoading();
        errorSection.classList.remove('hidden');
        errorText.textContent = msg;
        resultsSection.classList.add('hidden');
    }

    function hideError() {
        errorSection.classList.add('hidden');
    }

    function setVoiceState(state, text) {
        voiceMode = state;
        switch (state) {
            case 'RECORDING':
                voiceState.classList.remove('hidden');
                voicePulse.className = 'voice-pulse';
                voiceStateText.textContent = text || 'Recording… Click ■ to stop';
                micBtn.classList.add('recording');
                micIcon.classList.add('hidden');
                stopIcon.classList.remove('hidden');
                break;
            case 'TRANSCRIBING':
                voicePulse.className = 'voice-pulse transcribing';
                voiceStateText.textContent = text || 'Transcribing speech…';
                micBtn.classList.remove('recording');
                micIcon.classList.remove('hidden');
                stopIcon.classList.add('hidden');
                break;
            case 'PROCESSING':
                voicePulse.className = 'voice-pulse processing';
                voiceStateText.textContent = text || 'Retrieving answer…';
                break;
            case 'SUCCESS':
            case 'ERROR':
            case 'IDLE':
            default:
                voiceState.classList.add('hidden');
                micBtn.classList.remove('recording');
                micIcon.classList.remove('hidden');
                stopIcon.classList.add('hidden');
                voiceMode = 'IDLE';
                break;
        }
    }

    // ── Render Latency Table ──
    function renderLatency(latency) {
        latencyBody.innerHTML = '';

        // Component timings
        var components = [
            ['STT', latency.stt_ms],
            ['Query Processing', latency.query_processing_ms],
            ['Embedding', latency.embedding_ms],
            ['Retrieval', latency.retrieval_ms],
            ['Reranking', latency.reranking_ms],
            ['Generation', latency.generation_ms],
            ['Guardrails', latency.guardrails_ms],
        ];

        for (var i = 0; i < components.length; i++) {
            var name = components[i][0];
            var ms = components[i][1];
            if (ms === undefined || ms === null || ms === 0) continue;
            var tr = document.createElement('tr');
            tr.innerHTML = '<td>' + name + '</td><td>' + ms.toFixed(2) + ' ms</td>';
            latencyBody.appendChild(tr);
        }

        // Divider
        var divRow = document.createElement('tr');
        divRow.className = 'latency-divider';
        divRow.innerHTML = '<td colspan="2">PRIMARY METRICS</td>';
        latencyBody.appendChild(divRow);

        // Primary metrics
        var primaries = [
            ['STT Latency', latency.stt_latency_ms, 'latency-primary'],
            ['RAG Latency', latency.rag_latency_ms, 'latency-primary'],
            ['E2E Latency', latency.e2e_latency_ms, 'latency-total'],
        ];

        for (var j = 0; j < primaries.length; j++) {
            var pName = primaries[j][0];
            var pMs = primaries[j][1];
            var pClass = primaries[j][2];
            if (pMs === undefined || pMs === null) continue;
            var ptr = document.createElement('tr');
            ptr.className = pClass;
            ptr.innerHTML = '<td>' + pName + '</td><td>' + pMs.toFixed(2) + ' ms</td>';
            latencyBody.appendChild(ptr);
        }
    }

    // ── Render Sources ──
    function renderSources(sources) {
        sourcesGrid.innerHTML = '';

        if (!sources || sources.length === 0) {
            sourcesSection.classList.add('hidden');
            return;
        }

        sourcesSection.classList.remove('hidden');

        for (var i = 0; i < sources.length; i++) {
            var src = sources[i];
            var panel = document.createElement('div');
            panel.className = 'source-panel';

            // Detect language from metadata
            var lang = '';
            if (src.metadata) {
                lang = src.metadata.language || src.metadata.lang || '';
            }

            var headerHTML = '<div class="source-header">' +
                '<span class="source-rank">SOURCE ' + String(i + 1).padStart(2, '0') + '</span>';

            if (lang) {
                headerHTML += '<span class="source-lang">' + escapeHTML(lang) + '</span>';
            }

            headerHTML += '<span class="source-score">' + src.score.toFixed(3) + '</span>' +
                '</div>';

            panel.innerHTML = headerHTML + '<p class="source-text">' + escapeHTML(src.passage_text) + '</p>';
            sourcesGrid.appendChild(panel);
        }
    }

    // ── Render Full Response ──
    function renderResponse(data, isVoice) {
        hideLoading();
        hideError();

        // Show transcription for voice queries
        if (isVoice && data.transcription) {
            transcriptionCard.classList.remove('hidden');
            transcriptionText.textContent = data.transcription;
            queryInput.value = data.transcription;
        } else {
            transcriptionCard.classList.add('hidden');
        }

        // Answer
        answerText.textContent = data.answer || '';

        // Status
        if (data.status && data.status !== 'success') {
            answerStatus.classList.remove('hidden');
            answerStatus.textContent = 'Status: ' + data.status;
        } else {
            answerStatus.classList.add('hidden');
        }

        // Sources
        renderSources(data.sources);

        // Latency
        if (data.latency) {
            renderLatency(data.latency);
        }

        // Show results
        resultsSection.classList.remove('hidden');

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Text Query ──
    async function submitQuery(queryText) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'SEARCHING…';
        showLoading('Retrieving answer…');

        try {
            var res = await fetch(API_BASE + '/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: queryText }),
            });

            if (!res.ok) {
                var err = {};
                try { err = await res.json(); } catch (_) { /* ignore */ }
                throw new Error(err.detail || 'Request failed: ' + res.status);
            }

            var data = await res.json();
            renderResponse(data, false);
        } catch (err) {
            showError(err.message || 'Failed to process query');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'ASK →';
        }
    }

    // ── Voice Recording ──
    async function startRecording() {
        try {
            var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = function (e) {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = function () {
                stream.getTracks().forEach(function (track) { track.stop(); });
                processVoiceQuery();
            };

            mediaRecorder.start();
            setVoiceState('RECORDING');
        } catch (err) {
            showError('Microphone access denied or unavailable: ' + err.message);
            setVoiceState('IDLE');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
    }

    async function processVoiceQuery() {
        setVoiceState('TRANSCRIBING');
        showLoading('Transcribing speech…');

        var audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        var formData = new FormData();
        formData.append('file', audioBlob, 'speech.webm');

        try {
            setVoiceState('PROCESSING', 'Processing voice query…');
            loadingText.textContent = 'Retrieving answer…';

            var res = await fetch(API_BASE + '/api/voice-query', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                throw new Error('Voice query request failed: ' + res.status);
            }

            var data = await res.json();

            if (data.status === 'empty_transcription') {
                showError('No speech detected in the recording. Please try again.');
                setVoiceState('IDLE');
                return;
            }

            if (data.status === 'error' && !data.answer) {
                showError(data.error || 'Voice query processing error');
                setVoiceState('IDLE');
                return;
            }

            setVoiceState('IDLE');
            renderResponse(data, true);

        } catch (err) {
            showError('Voice query failed: ' + err.message);
            setVoiceState('IDLE');
        }
    }

    // ── Utility ──
    function escapeHTML(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Event Listeners ──
    queryForm.addEventListener('submit', function (e) {
        e.preventDefault();
        var q = queryInput.value.trim();
        if (q) submitQuery(q);
    });

    micBtn.addEventListener('click', function () {
        if (voiceMode === 'IDLE') {
            startRecording();
        } else if (voiceMode === 'RECORDING') {
            stopRecording();
        }
    });

    errorDismiss.addEventListener('click', function () {
        hideError();
    });

    // ── Initialize ──
    checkHealth();

    // Re-check health every 30 seconds
    setInterval(checkHealth, 30000);
})();
