(function () {
    'use strict';

    const API_BASE = window.location.origin;

    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const submitBtn = document.getElementById('submit-btn');
    const responseSection = document.getElementById('response-section');
    const answerText = document.getElementById('answer-text');
    const sourcesCard = document.getElementById('sources-card');
    const sourcesList = document.getElementById('sources-list');
    const latencyBody = document.getElementById('latency-body');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');

    async function checkHealth() {
        try {
            statusIndicator.className = 'status checking';
            statusText.textContent = 'Checking...';
            const res = await fetch(API_BASE + '/health');
            if (res.ok) {
                const data = await res.json();
                statusIndicator.className = 'status connected';
                statusText.textContent = data.app_name + ' v' + data.version;
            } else {
                throw new Error('Health check failed');
            }
        } catch {
            statusIndicator.className = 'status disconnected';
            statusText.textContent = 'Disconnected';
        }
    }

    function showError(msg) {
        errorSection.classList.remove('hidden');
        errorText.textContent = msg;
        responseSection.classList.add('hidden');
    }

    function renderLatency(latency) {
        latencyBody.innerHTML = '';
        const fields = [
            ['STT Processing', latency.stt_ms],
            ['Query Processing', latency.query_processing_ms],
            ['Embedding', latency.embedding_ms],
            ['Retrieval', latency.retrieval_ms],
            ['Reranking', latency.reranking_ms],
            ['Generation', latency.generation_ms],
            ['Guardrails', latency.guardrails_ms],
            ['--- Primary Metrics ---', null],
            ['STT Latency (isolated)', latency.stt_latency_ms ?? latency.stt_ms],
            ['RAG Latency', latency.rag_latency_ms],
            ['Full E2E Latency (STT + RAG)', latency.e2e_latency_ms],
            ['Total Request (server)', latency.total_request_ms],
        ];
        for (const [name, ms] of fields) {
            const tr = document.createElement('tr');
            if (ms === null) {
                tr.innerHTML = '<td colspan="2" style="font-weight:bold; color:#888; padding-top:8px; border-top:1px solid #333;">' + name + '</td>';
            } else {
                const isSummary = name.includes('Latency') || name.includes('Total');
                tr.innerHTML =
                    '<td' + (isSummary ? ' style="font-weight:bold; color:#7986cb;"' : '') + '>' + name + '</td>' +
                    '<td' + (isSummary ? ' style="font-weight:bold; color:#7986cb;"' : '') + '>' + (ms !== undefined ? ms.toFixed(2) : '—') + '</td>';
            }
            latencyBody.appendChild(tr);
        }
    }

    async function submitQuery(queryText) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Searching...';
        errorSection.classList.add('hidden');

        try {
            const res = await fetch(API_BASE + '/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: queryText }),
            });

            if (!res.ok) {
                const err = await res.json().catch(function () { return {}; });
                throw new Error(err.detail || 'Request failed: ' + res.status);
            }

            const data = await res.json();

            responseSection.classList.remove('hidden');
            answerText.textContent = data.answer;

            if (data.sources && data.sources.length > 0) {
                sourcesCard.classList.remove('hidden');
                sourcesList.innerHTML = '';
                for (const src of data.sources) {
                    const li = document.createElement('li');
                    li.textContent = src.passage_text + ' (score: ' + src.score.toFixed(3) + ')';
                    sourcesList.appendChild(li);
                }
            } else {
                sourcesCard.classList.add('hidden');
            }

            renderLatency(data.latency);
        } catch (err) {
            showError(err.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Search';
        }
    }

    // Voice Input Handling (Phase 9)
    const micBtn = document.getElementById('mic-btn');
    const micStatus = document.getElementById('mic-status');
    const micStatusText = document.getElementById('mic-status-text');
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    if (micBtn) {
        micBtn.addEventListener('click', async function () {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = function (e) {
                        if (e.data.size > 0) audioChunks.push(e.data);
                    };

                    mediaRecorder.onstop = async function () {
                        stream.getTracks().forEach(function (track) { track.stop(); });
                        micStatusText.textContent = 'Processing voice query...';
                        micBtn.className = 'mic-idle';
                        micBtn.textContent = '🎤';

                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const formData = new FormData();
                        formData.append('file', audioBlob, 'speech.webm');

                        try {
                            const res = await fetch(API_BASE + '/api/voice-query', {
                                method: 'POST',
                                body: formData,
                            });
                            if (!res.ok) throw new Error('Voice query request failed');
                            const data = await res.json();
                            if (data.status === 'empty_transcription') {
                                showError('No speech detected in audio.');
                            } else if (data.status === 'error' && !data.answer) {
                                showError(data.error || 'Voice query processing error');
                            } else {
                                queryInput.value = data.transcription || '';
                                responseSection.classList.remove('hidden');
                                errorSection.classList.add('hidden');
                                answerText.textContent = data.answer;

                                if (data.sources && data.sources.length > 0) {
                                    sourcesCard.classList.remove('hidden');
                                    sourcesList.innerHTML = '';
                                    for (const src of data.sources) {
                                        const li = document.createElement('li');
                                        li.textContent = src.passage_text + ' (score: ' + src.score.toFixed(3) + ')';
                                        sourcesList.appendChild(li);
                                    }
                                } else {
                                    sourcesCard.classList.add('hidden');
                                }

                                renderLatency(data.latency);
                                micStatusText.textContent = 'Completed in ' + (data.latency.e2e_latency_ms || 0).toFixed(1) + ' ms';
                                setTimeout(function () { micStatus.classList.add('hidden'); }, 3000);
                            }
                        } catch (err) {
                            showError('Voice query failed: ' + err.message);
                            micStatus.classList.add('hidden');
                        }
                    };


                    mediaRecorder.start();
                    isRecording = true;
                    micBtn.className = 'mic-recording';
                    micBtn.textContent = '⏹️';
                    micStatus.classList.remove('hidden');
                    micStatusText.textContent = 'Recording speech... Click ⏹️ to finish';
                } catch (err) {
                    showError('Microphone access denied or unavailable: ' + err.message);
                }
            } else {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                isRecording = false;
            }
        });
    }

    queryForm.addEventListener('submit', function (e) {
        e.preventDefault();
        var q = queryInput.value.trim();
        if (q) submitQuery(q);
    });

    checkHealth();
})();

