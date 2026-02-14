document.addEventListener('DOMContentLoaded', () => {
    // --- Navigation ---
    const tabs = document.querySelectorAll('.nav-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            // Add active class
            tab.classList.add('active');
            const target = tab.getAttribute('data-tab');
            document.getElementById(target).classList.add('active');

            // Refresh History if selected
            if (target === 'history') {
                loadHistory();
            }

        });
    });

    // --- Tab 1: Single Prediction ---
    const form = document.getElementById('predictionForm');
    const resultCard = document.getElementById('resultCard');
    const closeResult = document.getElementById('closeResult');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Type conversion
        data.MonthlyCharges = parseFloat(data.MonthlyCharges);
        data.tenure = parseFloat(data.tenure);
        data.total_services = parseInt(data.total_services || 1);

        try {
            const btn = form.querySelector('button');
            const originalText = btn.innerText;
            btn.innerText = "Analyzing...";

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await res.json();

            if (res.ok) {
                showResult(result);
            } else {
                alert(result.error);
            }
            btn.innerText = originalText;
        } catch (err) {
            console.error(err);
            alert('Server Error');
        }
    });

    function showResult(data) {
        resultCard.classList.remove('hidden');

        // Probability
        const percent = Math.round(data.churn_probability * 100);
        document.getElementById('probValue').innerText = percent;

        // Risk Badge
        const badge = document.getElementById('riskBadge');
        badge.innerText = `${data.risk_level} Risk`;

        // Color Coding
        const circle = document.querySelector('.score-circle');
        if (data.churn_probability > 0.5) {
            circle.style.borderColor = '#EF4444'; // Red
            badge.style.backgroundColor = '#FEE2E2';
            badge.style.color = '#B91C1C';
        } else if (data.churn_probability > 0.3) {
            circle.style.borderColor = '#F59E0B'; // Amber
            badge.style.backgroundColor = '#FEF3C7';
            badge.style.color = '#D97706';
        } else {
            circle.style.borderColor = '#10B981'; // Green
            badge.style.backgroundColor = '#D1FAE5';
            badge.style.color = '#047857';
        }

        // Factors (Premium UI)
        updateInsightUI(data);
    }

    closeResult.addEventListener('click', () => {
        resultCard.classList.add('hidden');
    });

    // --- Tab 2: Batch Upload ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const statusMsg = document.getElementById('uploadStatus');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length) handleUpload(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleUpload(fileInput.files[0]);
    });

    async function handleUpload(file) {
        statusMsg.innerText = `Uploading ${file.name}...`;
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const result = await res.json();

            if (res.ok) {
                statusMsg.innerText = `Success: Processed ${result.results.length} records. Check History tab.`;
                statusMsg.style.color = '#059669'; // Emerald 600
            } else {
                statusMsg.innerText = `Error: ${result.error}`;
                statusMsg.style.color = '#DC2626'; // Red 600
            }
        } catch (err) {
            statusMsg.innerText = `Network Error`;
            statusMsg.style.color = '#DC2626';
        }
    }

    // --- Tab 3: History ---
    const historyTable = document.getElementById('historyTable').querySelector('tbody');
    const clearBtn = document.getElementById('clearHistory');

    // Update table header first to include ID column
    const thead = document.getElementById('historyTable').querySelector('thead tr');
    if (!thead.innerHTML.includes('ID')) {
        thead.insertCell(0).outerHTML = "<th>ID</th>";
    }

    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();

            historyTable.innerHTML = '';

            data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${row.input_data.customerID || row.input_data.CustomerID || '-'}</strong></td>
                    <td>${row.input_data.Contract || '-'}</td>
                    <td>${row.input_data.tenure || 0} mos</td>
                    <td>$${row.input_data.MonthlyCharges || 0}</td>
                    <td><span class="badge ${row.risk_level.toLowerCase()}">${row.risk_level}</span></td>
                    <td>${Math.round(row.churn_probability * 100)}%</td>
                `;
                historyTable.appendChild(tr);
            });
        } catch (err) {
            console.error('Failed to load history');
        }
    }

    clearBtn.addEventListener('click', async () => {
        if (!confirm('Clear session history?')) return;

        await fetch('/api/history', { method: 'DELETE' });
        loadHistory();
    });

    // --- Psychology Slide-Panel ---
    const psyPanel = document.getElementById('psyPanel');
    const psyOverlay = document.getElementById('psyOverlay');
    const openPsyBtn = document.getElementById('openPsy');
    const closePsyBtn = document.getElementById('closePsy');

    function togglePanel(show) {
        if (show) {
            psyPanel.classList.remove('hidden-panel');
            psyOverlay.classList.remove('hidden-panel');
        } else {
            psyPanel.classList.add('hidden-panel');
            psyOverlay.classList.add('hidden-panel');
        }
    }

    if (openPsyBtn && psyPanel) {
        openPsyBtn.addEventListener('click', () => togglePanel(true));
        closePsyBtn.addEventListener('click', () => togglePanel(false));
        psyOverlay.addEventListener('click', () => togglePanel(false));
    }
});

// Helper for Premium Insight Cards
function updateInsightUI(data) {
    const list = document.getElementById('factorsList');
    list.innerHTML = '';

    if (data.causal_insights && data.causal_insights.length > 0) {
        data.causal_insights.forEach(insight => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            card.innerHTML = `
                <div class="insight-header">
                    <div class="icon-shape"></div>
                    <span>${insight.description}</span>
                </div>
                <div class="insight-action">
                    ACTION: ${insight.action}
                </div>
            `;
            list.appendChild(card);
        });
    } else {
        list.innerHTML = '<div style="color:#6B7280; font-style:italic;">No specific risk factors identified.</div>';
    }
}
