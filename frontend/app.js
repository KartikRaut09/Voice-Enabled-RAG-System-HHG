/**
 * HH GOA 2026 — Voice RAG Application Logic
 * Integrates text query, Sarvam voice recording, comprehensive RAG metrics,
 * system diagnostics, and verified performance benchmarks.
 */
(function () {
    'use strict';

    const API_BASE = window.location.origin;

    // DOM Elements
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const micBtn = document.getElementById('mic-btn');
    const micIcon = document.getElementById('mic-icon');
    const stopIcon = document.getElementById('stop-icon');
    const voiceState = document.getElementById('voice-state');
    const voiceDot = document.getElementById('voice-dot');
    const voiceStateText = document.getElementById('voice-state-text');

    const loadingSection = document.getElementById('loading-section');
    const loadingText = document.getElementById('loading-text');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');
    const errorDismiss = document.getElementById('error-dismiss');

    const resultsSection = document.getElementById('results-section');
    const queryEcho = document.getElementById('query-echo');
    const queryEchoText = document.getElementById('query-echo-text');
    const transcriptionRow = document.getElementById('transcription-row');
    const transcriptionText = document.getElementById('transcription-text');
    const answerText = document.getElementById('answer-text');
    const answerStatus = document.getElementById('answer-status');
    const sourcesRow = document.getElementById('sources-row');
    const sourcesCount = document.getElementById('sources-count');
    const sourcesList = document.getElementById('sources-list');
    const perfBody = document.getElementById('perf-body');
    const systemGrid = document.getElementById('system-grid');
    const benchmarksSection = document.getElementById('benchmarks-section');

    // Voice Recording State
    let mediaRecorder = null;
    let audioChunks = [];
    let voiceMode = 'IDLE'; // IDLE | RECORDING | TRANSCRIBING | PROCESSING | SUCCESS | ERROR

    // ── Health Check ──
    async function checkHealth() {
        try {
            statusIndicator.className = 'status-badge checking';
            statusText.textContent = 'Checking…';
            const res = await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(4000) });
            if (res.ok) {
                const data = await res.json();
                statusIndicator.className = 'status-badge connected';
                statusText.textContent = 'SYSTEM LIVE';
            } else {
                throw new Error('Health probe failed');
            }
        } catch {
            statusIndicator.className = 'status-badge disconnected';
            statusText.textContent = 'DISCONNECTED';
        }
    }

    // ── UI Visibility Helpers ──
    function showLoading(msg) {
        loadingText.textContent = msg || 'Processing query…';
        loadingSection.classList.remove('hidden');
        errorSection.classList.add('hidden');
    }

    function hideLoading() {
        loadingSection.classList.add('hidden');
    }

    function showError(msg) {
        hideLoading();
        errorText.textContent = msg;
        errorSection.classList.remove('hidden');
    }

    function hideError() {
        errorSection.classList.add('hidden');
    }

    function setVoiceState(state, text) {
        voiceMode = state;
        switch (state) {
            case 'RECORDING':
                voiceState.classList.remove('hidden');
                voiceDot.className = 'voice-dot';
                voiceStateText.textContent = text || 'Recording… Click ■ to stop';
                micBtn.classList.add('recording');
                micIcon.classList.add('hidden');
                stopIcon.classList.remove('hidden');
                break;
            case 'TRANSCRIBING':
                voiceDot.className = 'voice-dot transcribing';
                voiceStateText.textContent = text || 'Transcribing with Sarvam saaras:v3…';
                micBtn.classList.remove('recording');
                micIcon.classList.remove('hidden');
                stopIcon.classList.add('hidden');
                break;
            case 'PROCESSING':
                voiceDot.className = 'voice-dot processing';
                voiceStateText.textContent = text || 'Executing Hybrid RAG Pipeline…';
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

    // ── Performance Table Renderer ──
    function renderPerformance(latency) {
        perfBody.innerHTML = '';
        if (!latency) return;

        const stages = [
            ['STT Inference', latency.stt_ms],
            ['Query Processing', latency.query_processing_ms],
            ['Embedding (Multilingual E5)', latency.embedding_ms],
            ['Hybrid Retrieval (FAISS + BM25)', latency.retrieval_ms],
            ['RRF Fusion & Context', latency.reranking_ms],
            ['LLM Generation (Groq LLaMA-3.1)', latency.generation_ms],
            ['Guardrails Validation', latency.guardrails_ms],
        ];

        stages.forEach(function (stage) {
            const name = stage[0];
            const ms = stage[1];
            if (ms !== undefined && ms !== null && ms > 0) {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td>' + name + '</td><td>' + ms.toFixed(2) + ' ms</td>';
                perfBody.appendChild(tr);
            }
        });

        // Primary highlights
        if (latency.rag_latency_ms) {
            const trRag = document.createElement('tr');
            trRag.className = 'perf-highlight';
            trRag.innerHTML = '<td>RAG Pipeline Latency</td><td>' + latency.rag_latency_ms.toFixed(2) + ' ms</td>';
            perfBody.appendChild(trRag);
        }

        if (latency.stt_latency_ms) {
            const trStt = document.createElement('tr');
            trStt.className = 'perf-highlight';
            trStt.innerHTML = '<td>STT Latency (Sarvam)</td><td>' + latency.stt_latency_ms.toFixed(2) + ' ms</td>';
            perfBody.appendChild(trStt);
        }

        const e2e = latency.e2e_latency_ms || latency.total_request_ms || latency.rag_latency_ms;
        if (e2e) {
            const trE2e = document.createElement('tr');
            trE2e.className = 'perf-total';
            trE2e.innerHTML = '<td>Full E2E Latency</td><td>' + e2e.toFixed(2) + ' ms</td>';
            perfBody.appendChild(trE2e);
        }
    }

    // ── Sources Renderer ──
    function renderSources(sources) {
        sourcesList.innerHTML = '';
        if (!sources || sources.length === 0) {
            sourcesRow.classList.add('hidden');
            return;
        }

        sourcesRow.classList.remove('hidden');
        sourcesCount.textContent = String(sources.length).padStart(2, '0') + ' SOURCES RETRIEVED';

        sources.forEach(function (src, idx) {
            const item = document.createElement('div');
            item.className = 'source-item';

            const lang = (src.metadata && (src.metadata.language || src.metadata.lang)) || 'INDIC';
            const score = typeof src.score === 'number' ? src.score.toFixed(3) : '—';

            item.innerHTML =
                '<div class="source-top">' +
                    '<span class="source-tag">[' + String(idx + 1).padStart(2, '0') + ']</span>' +
                    '<span class="source-lang-tag">' + escapeHTML(lang) + '</span>' +
                    '<span class="source-score-val">Score ' + score + '</span>' +
                '</div>' +
                '<p class="source-body">' + escapeHTML(src.passage_text || '') + '</p>';

            sourcesList.appendChild(item);
        });
    }

    // ── System Diagnostics Renderer ──
    function renderSystemDiagnostics(data) {
        systemGrid.innerHTML = '';

        const pipeMeta = data.pipeline_metadata || {};
        const qMeta = data.query_metadata || {};
        const guard = data.guardrail_flags || {};

        const stats = [
            ['STT Provider', 'Sarvam AI (saaras:v3)'],
            ['LLM Provider', (pipeMeta.provider || 'Groq') + ' · ' + (pipeMeta.model_used || 'llama-3.1-8b-instant')],
            ['Retrieval Mode', (pipeMeta.retrieval_mode || 'Hybrid (FAISS + BM25 + RRF)').toUpperCase()],
            ['Detected Lang', (qMeta.language || data.language || 'Auto-Detected').toUpperCase()],
            ['Tokens (In/Out)', (pipeMeta.input_tokens || '—') + ' / ' + (pipeMeta.output_tokens || '—')],
            ['Guardrails', guard.passed === false ? 'BLOCKED' : 'PASS (Citations & Grounding Active)'],
        ];

        stats.forEach(function (pair) {
            const div = document.createElement('div');
            div.className = 'system-stat';
            div.innerHTML = '<span class="stat-key">' + pair[0] + '</span><span class="stat-val">' + escapeHTML(String(pair[1])) + '</span>';
            systemGrid.appendChild(div);
        });
    }

    // ── Response Orchestrator ──
    function renderResponse(data, originalQuery, isVoice) {
        hideLoading();
        hideError();

        // 1. Query Echo
        if (originalQuery || data.query) {
            queryEcho.classList.remove('hidden');
            queryEchoText.textContent = originalQuery || data.query;
        }

        // 2. Transcription (Voice Query)
        if (isVoice && data.transcription) {
            transcriptionRow.classList.remove('hidden');
            transcriptionText.textContent = data.transcription;
            queryInput.value = data.transcription;
        } else {
            transcriptionRow.classList.add('hidden');
        }

        // 3. Answer
        answerText.textContent = data.answer || 'No response generated.';
        if (data.status && data.status !== 'success') {
            answerStatus.classList.remove('hidden');
            answerStatus.textContent = 'Status: ' + data.status;
        } else {
            answerStatus.classList.add('hidden');
        }

        // 4. Sources
        renderSources(data.sources);

        // 5. Performance
        renderPerformance(data.latency);

        // 6. System Diagnostics
        renderSystemDiagnostics(data);

        // 7. Show Results & Benchmarks
        resultsSection.classList.remove('hidden');
        benchmarksSection.classList.remove('hidden');

        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ── Text Query Dispatch ──
    async function submitTextQuery(text) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'ASKING…';
        showLoading('Retrieving grounded answer…');

        try {
            const res = await fetch(API_BASE + '/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text }),
            });

            if (!res.ok) {
                let errData = {};
                try { errData = await res.json(); } catch (_) {}
                throw new Error(errData.detail || 'Request failed with status ' + res.status);
            }

            const data = await res.json();
            renderResponse(data, text, false);
        } catch (err) {
            showError(err.message || 'Failed to process query');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'ASK →';
        }
    }

    // ── Voice Query Dispatch ──
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = function (e) {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = function () {
                stream.getTracks().forEach(function (track) { track.stop(); });
                processVoiceAudio();
            };

            mediaRecorder.start();
            setVoiceState('RECORDING');
        } catch (err) {
            showError('Microphone access unavailable: ' + err.message);
            setVoiceState('IDLE');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
    }

    async function processVoiceAudio() {
        setVoiceState('TRANSCRIBING');
        showLoading('Transcribing Indic voice with Sarvam AI…');

        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'speech.webm');

        try {
            setVoiceState('PROCESSING', 'Orchestrating RAG Pipeline…');
            loadingText.textContent = 'Retrieving grounded answer…';

            const res = await fetch(API_BASE + '/api/voice-query', {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                throw new Error('Voice query request failed: ' + res.status);
            }

            const data = await res.json();

            if (data.status === 'empty_transcription') {
                showError('No speech detected in the audio payload. Please speak clearly and try again.');
                setVoiceState('IDLE');
                return;
            }

            if (data.status === 'error' && !data.answer) {
                showError(data.error || 'Voice query processing error');
                setVoiceState('IDLE');
                return;
            }

            setVoiceState('IDLE');
            renderResponse(data, data.transcription, true);

        } catch (err) {
            showError('Voice query failed: ' + err.message);
            setVoiceState('IDLE');
        }
    }

    // ── Utilities ──
    function escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Event Handlers ──
    queryForm.addEventListener('submit', function (e) {
        e.preventDefault();
        const q = queryInput.value.trim();
        if (q) submitTextQuery(q);
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

    // ── Init ──
    checkHealth();
    setInterval(checkHealth, 25000);
})();
