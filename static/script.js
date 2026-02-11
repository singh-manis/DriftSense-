let reconstructionChartInstance = null;
let histogramChartInstance = null;
let modalChartInstance = null;
let networkInstance = null; // For Vis.js Control Flow Graph
let globalData = null; // Stores dataset for access in modal
let currentDriftIndex = null; // To track which index we are looking at

// --- 1. UPLOAD & ANALYZE ---
async function uploadAndAnalyze() {
    const fileInput = document.getElementById('csvFile');
    if (!fileInput.files[0]) {
        alert("Please select a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show loading state
    const btn = document.querySelector('.btn-primary');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.error) {
            alert(data.error);
        } else {
            globalData = data; // Save data globally
            updateDashboard(data);
        }
    } catch (error) {
        console.error('Error:', error);
        alert("An error occurred during analysis:\n" + error.message);
    } finally {
        btn.innerText = originalText;
    }
}

// --- 2. UPDATE DASHBOARD (Stats + Graphs) ---
function updateDashboard(data) {
    // A. Update Statistics Cards
    document.getElementById('totalTraces').innerText = data.total_traces;
    document.getElementById('driftCount').innerText = data.drift_points_count;

    // B. Update Drift Summary Row
    if (document.getElementById('stat-threshold')) {
        document.getElementById('stat-threshold').innerText = Number(data.threshold).toFixed(5);
        document.getElementById('stat-mode').innerText = data.mode || 'N/A';
        document.getElementById('stat-engine').innerText = data.engine || 'N/A';
        document.getElementById('stat-max').innerText = Number(data.max_error).toFixed(5);
        document.getElementById('stat-mean').innerText = Number(data.mean_error).toFixed(5);
        document.getElementById('stat-std').innerText = Number(data.std_dev).toFixed(5);

        // Added Precision/Recall
        if (document.getElementById('stat-precision')) {
            document.getElementById('stat-precision').innerText = Number(data.precision).toFixed(4);
        }
        if (document.getElementById('stat-recall')) {
            document.getElementById('stat-recall').innerText = data.recall;
        }
    }

    // C. Render Index Badges (The blue numbers)
    renderIndexBadges(data);

    // D. Render MAIN Charts (This is the part you were missing)
    renderMainCharts(data);
}

function renderIndexBadges(data) {
    const container = document.getElementById('indices-container');
    const toggleBtn = document.getElementById('toggleIndicesBtn');

    if (!container) return; // Guard clause if element missing

    container.innerHTML = '';

    if (data.drift_indices.length === 0) {
        container.innerHTML = '<span style="color:#94a3b8; font-style:italic;">No drift detected.</span>';
        if (toggleBtn) toggleBtn.style.display = 'none';
    } else {
        data.drift_indices.forEach(index => {
            const badge = document.createElement('div');
            badge.className = 'index-badge';
            badge.innerText = index;
            // Click event to open the Pop-up Modal
            badge.onclick = () => openDriftModal(index);
            container.appendChild(badge);
        });

        if (toggleBtn) {
            if (data.drift_indices.length > 30) {
                toggleBtn.style.display = 'inline-block';
                toggleBtn.innerText = "Show More";
                container.classList.add('collapsed');
            } else {
                toggleBtn.style.display = 'none';
                container.classList.remove('collapsed');
            }
        }
    }
}

// --- 3. RENDER MAIN CHARTS (Line & Histogram) ---
function renderMainCharts(data) {
    // --- Chart 1: Reconstruction Error (Line Chart) ---
    const ctxLine = document.getElementById('reconstructionChart').getContext('2d');
    if (reconstructionChartInstance) reconstructionChartInstance.destroy();

    const labels = Array.from({ length: data.total_traces }, (_, i) => i);

    // ... inside renderMainCharts ...

    reconstructionChartInstance = new Chart(ctxLine, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Reconstruction Error',
                data: data.reconstruction_errors,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                pointRadius: 2,
                pointHoverRadius: 6,
                tension: 0
            },
            {
                label: 'Threshold',
                data: Array(data.total_traces).fill(data.threshold),
                borderColor: '#ef4444',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) label += context.parsed.y.toFixed(4);
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    grid: { color: '#1e293b' },
                    ticks: { color: '#94a3b8' },
                    // --- ADDED: Y-Axis Label ---
                    title: {
                        display: true,
                        text: 'Reconstruction Error (MSE)',
                        color: '#94a3b8',
                        font: { size: 12, weight: 'bold' }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' },
                    // --- ADDED: X-Axis Label ---
                    title: {
                        display: true,
                        text: 'Trace Index (Time)',
                        color: '#94a3b8',
                        font: { size: 12, weight: 'bold' }
                    }
                }
            }
        }
    });

    // --- Chart 2: Histogram (Bar Chart) ---
    const ctxHist = document.getElementById('histogramChart').getContext('2d');
    if (histogramChartInstance) histogramChartInstance.destroy();

    // Simple binning logic for histogram
    const errors = data.reconstruction_errors;
    const binCount = 20;
    const maxErr = Math.max(...errors) || 1;
    const bins = new Array(binCount).fill(0);

    errors.forEach(e => {
        const binIndex = Math.min(Math.floor((e / maxErr) * binCount), binCount - 1);
        bins[binIndex]++;
    });

    histogramChartInstance = new Chart(ctxHist, {
        type: 'bar',
        data: {
            labels: Array.from({ length: binCount }, (_, i) => (i * (maxErr / binCount)).toFixed(2)),
            datasets: [{
                label: 'Frequency',
                data: bins,
                backgroundColor: '#8b5cf6' // Purple
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    grid: { color: '#1e293b' },
                    ticks: { color: '#94a3b8' },
                    // ADDED: Y-Axis Label
                    title: {
                        display: true,
                        text: 'Frequency (Count)',
                        color: '#94a3b8',
                        font: { size: 12 }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        display: true, // Changed to true so you can see the error values
                        color: '#94a3b8',
                        maxTicksLimit: 10 // Limit labels so they don't overlap
                    },
                    // ADDED: X-Axis Label
                    title: {
                        display: true,
                        text: 'Reconstruction Error Range',
                        color: '#94a3b8',
                        font: { size: 12 }
                    }
                }
            }
        }
    });
}

// --- 4. MODAL & CONTROL FLOW GRAPH ---
function openDriftModal(targetIndex) {
    // --- CRITICAL FIX: Update the global variable ---
    currentDriftIndex = targetIndex;
    console.log("Drift Index Updated to:", currentDriftIndex); // Debugging check

    const modal = document.getElementById('driftModal');
    const title = document.getElementById('modalTitle');

    // Reset Gemini Box so it doesn't show old answers
    const geminiBox = document.getElementById('geminiResponse');
    if (geminiBox) {
        geminiBox.style.display = 'none';
        geminiBox.innerHTML = '<div class="typing-indicator">Gemini is thinking...</div>';
    }

    // Define Window
    const windowSize = 5;
    const start = Math.max(0, targetIndex - windowSize);
    const end = Math.min(globalData.total_traces, targetIndex + windowSize);

    const localLabels = Array.from({ length: end - start }, (_, i) => start + i);
    const localErrors = globalData.reconstruction_errors.slice(start, end);
    const localThreshold = Array(localLabels.length).fill(globalData.threshold);

    title.innerText = `Drift Point Analysis: Index ${targetIndex}`;
    modal.style.display = "flex"; // Keep your centering fix

    // Render Line Chart
    const ctx = document.getElementById('modalChart').getContext('2d');
    if (modalChartInstance) modalChartInstance.destroy();

    modalChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: localLabels,
            datasets: [{
                label: 'Local Error',
                data: localErrors,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.2)',
                borderWidth: 2,
                pointRadius: 4,
                tension: 0.1
            }, {
                label: 'Threshold',
                data: localThreshold,
                borderColor: '#ef4444',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: '#2f334d' }, ticks: { color: '#94a3b8' } },
                x: { grid: { color: '#2f334d' }, ticks: { color: '#94a3b8' } }
            }
        }
    });

    // Render Control Flow Graph
    renderControlFlowGraph(targetIndex);

    // Render 3D Latent Space (Wait for modal transition)
    if (globalData.latent_space) {
        setTimeout(() => {
            renderLatentChart(globalData.latent_space, targetIndex);
            Plotly.Plots.resize('latent3dChart');
        }, 200);
    }
}

function renderControlFlowGraph(traceIndex) {
    const container = document.getElementById('cfgNetwork');

    // Safety check: Ensure backend sent feature importance
    if (!globalData || !globalData.feature_importance || !globalData.column_names) {
        container.innerHTML = '<p style="color:#94a3b8; padding:20px; text-align:center;">No feature data available for graph.</p>';
        return;
    }

    // Get feature values (errors) for this specific trace
    const traceValues = globalData.feature_importance[traceIndex];
    const activities = globalData.column_names;

    // Sort activities by importance (highest error first)
    // This ensures we always show the "most active/drifted" nodes
    let rankedActivities = activities.map((name, i) => ({
        index: i,
        label: name,
        value: traceValues[i] || 0
    }));

    rankedActivities.sort((a, b) => b.value - a.value);

    // Pick Top 5-7 activities (Guarantees graph is never empty)
    // Filter out 0s just in case, but take at least top ones
    const topActivities = rankedActivities.filter(a => a.value > 0).slice(0, 7);

    // Build Vis.js Nodes and Edges
    const nodes = [];
    const edges = [];
    let prevId = 'start';

    nodes.push({ id: 'start', label: 'Start', color: '#2ac3de', shape: 'circle', font: { color: 'white' } });

    if (topActivities.length === 0) {
        // Fallback for empty/zero trace
        nodes.push({ id: 'none', label: 'No Significant\nActivity', shape: 'box', color: '#1e293b', font: { color: '#94a3b8' } });
        edges.push({ from: 'start', to: 'none', arrows: 'to', color: { color: '#565f89' } });
    } else {
        topActivities.forEach(item => {
            nodes.push({
                id: item.index,
                label: `${item.label}\n(Err: ${item.value.toFixed(4)})`,
                color: '#1e293b',
                font: { color: '#94a3b8' },
                shape: 'box',
                borderWidth: 2
            });
            edges.push({ from: prevId, to: item.index, arrows: 'to', color: { color: '#565f89' } });
            prevId = item.index;
        });
    }

    nodes.push({ id: 'end', label: 'End', color: '#f7768e', shape: 'circle', font: { color: 'white' } });
    edges.push({ from: prevId, to: 'end', arrows: 'to', color: { color: '#565f89' } });

    // Render Network
    const data = {
        nodes: new vis.DataSet(nodes),
        edges: new vis.DataSet(edges)
    };

    const options = {
        layout: {
            hierarchical: {
                direction: 'UD',
                sortMethod: 'directed',
                levelSpacing: 100
            }
        },
        physics: false,
        interaction: { dragNodes: true, zoomView: true }
    };

    if (networkInstance) networkInstance.destroy();
    networkInstance = new vis.Network(container, data, options);
}

// --- 5. HELPER FUNCTIONS ---
function closeModal() {
    document.getElementById('driftModal').style.display = "none";
}

function toggleIndices() {
    const container = document.getElementById('indices-container');
    const btn = document.getElementById('toggleIndicesBtn');

    container.classList.toggle('collapsed');

    if (container.classList.contains('collapsed')) {
        btn.innerText = "Show More";
    } else {
        btn.innerText = "Show Less";
    }
}

// Navigation (Sidebar switching)
function showSection(sectionId, element) {
    // Hide all sections
    const sections = ['dashboard', 'guide', 'datasets', 'team', 'version'];
    sections.forEach(id => {
        const el = document.getElementById(id + '-section');
        if (el) el.style.display = 'none';
    });

    // Show target section
    const target = document.getElementById(sectionId + '-section');
    if (target) target.style.display = 'block';

    // Update Sidebar Active Class
    const navItems = document.querySelectorAll('.nav-links li');
    navItems.forEach(item => item.classList.remove('active'));
    if (element) element.classList.add('active');
}
// --- NEW: GEMINI AI FUNCTION ---
async function askGemini() {
    if (currentDriftIndex === null) return;

    const responseBox = document.getElementById('geminiResponse');
    responseBox.style.display = 'block';

    // 1. Prepare Data
    // Re-calculate top features (same logic as the graph)
    const traceValues = globalData.feature_importance[currentDriftIndex];
    const activities = globalData.column_names;

    let rankedActivities = activities.map((name, i) => ({
        label: name,
        value: traceValues[i] || 0
    }));
    rankedActivities.sort((a, b) => b.value - a.value);
    const topFeatures = rankedActivities.filter(a => a.value > 0).slice(0, 5);

    const payload = {
        trace_index: currentDriftIndex,
        error_val: globalData.reconstruction_errors[currentDriftIndex].toFixed(5),
        threshold: globalData.threshold.toFixed(5),
        top_features: topFeatures
    };

    try {
        // 2. Call Backend
        const res = await fetch('/explain_drift', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        // 3. Display Result
        // Simple formatting: Convert **text** to bold
        let formattedText = data.explanation.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/\n/g, '<br>'); // Newlines

        responseBox.innerHTML = formattedText;

    } catch (error) {
        responseBox.innerHTML = `<span style="color: #f7768e;">Error: ${error.message}</span>`;
    }
}
// --- THEME TOGGLE LOGIC ---

// 1. Check LocalStorage on Load
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    const body = document.body;
    const icon = document.getElementById('themeIcon');

    if (savedTheme === 'light') {
        body.setAttribute('data-theme', 'light');
        if (icon) icon.innerText = '🌙'; // Show Moon icon (to switch back to dark)
    } else {
        body.removeAttribute('data-theme'); // Default to Dark
        if (icon) icon.innerText = '🌞'; // Show Sun icon
    }
});

// 2. Toggle Function
function toggleTheme() {
    const body = document.body;
    const icon = document.getElementById('themeIcon');
    const currentTheme = body.getAttribute('data-theme');

    if (currentTheme === 'light') {
        // Switch to Dark
        body.removeAttribute('data-theme');
        icon.innerText = '🌞';
        localStorage.setItem('theme', 'dark');
        updateChartColors(false); // Optional: Update charts for dark mode
    } else {
        // Switch to Light
        body.setAttribute('data-theme', 'light');
        icon.innerText = '🌙';
        localStorage.setItem('theme', 'light');
        updateChartColors(true); // Optional: Update charts for light mode
    }
}

// 3. Optional: Helper to update Chart.js Grid Lines
function updateChartColors(isLight) {
    const gridColor = isLight ? '#e5e7eb' : '#2f334d';
    const textColor = isLight ? '#64748b' : '#94a3b8';

    // Helper to safely update a chart instance if it exists
    const updateChart = (chart) => {
        if (chart) {
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.y.grid.color = gridColor;
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.y.ticks.color = textColor;
            chart.update();
        }
    };

    if (typeof reconstructionChartInstance !== 'undefined') updateChart(reconstructionChartInstance);
    if (typeof histogramChartInstance !== 'undefined') updateChart(histogramChartInstance);
    if (typeof modalChartInstance !== 'undefined') updateChart(modalChartInstance);
}


// --- 6. 3D LATENT SPACE VISUALIZATION (Plotly.js) ---
function renderLatentChart(latentData, targetIndex) {
    if (!latentData || latentData.length === 0) return;

    // Separate data by status for coloring
    const normalX = [], normalY = [], normalZ = [];
    const driftX = [], driftY = [], driftZ = [];
    const targetX = [], targetY = [], targetZ = [];

    latentData.forEach(p => {
        if (p.id === targetIndex) {
            targetX.push(p.x); targetY.push(p.y); targetZ.push(p.z);
        } else if (p.status === 1) { // Drift
            driftX.push(p.x); driftY.push(p.y); driftZ.push(p.z);
        } else { // Normal
            normalX.push(p.x); normalY.push(p.y); normalZ.push(p.z);
        }
    });

    const traceNormal = {
        x: normalX, y: normalY, z: normalZ,
        mode: 'markers',
        type: 'scatter3d',
        name: 'Normal Behavior',
        marker: { size: 3, color: '#3b82f6', opacity: 0.4 } // Blue
    };

    const traceDrift = {
        x: driftX, y: driftY, z: driftZ,
        mode: 'markers',
        type: 'scatter3d',
        name: 'Drift Detected',
        marker: { size: 4, color: '#ef4444', opacity: 0.8 } // Red
    };

    // Highlight the specific point the user clicked
    const traceTarget = {
        x: targetX, y: targetY, z: targetZ,
        mode: 'markers',
        type: 'scatter3d',
        name: 'Current Focus',
        marker: { size: 10, color: '#facc15', symbol: 'diamond', opacity: 1 } // Yellow diamond
    };

    const layout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        width: document.getElementById('latent3dChart').clientWidth, // Explicit width
        height: 400, // Explicit height
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        scene: {
            xaxis: {
                title: 'Component 1',
                gridcolor: '#2f334d', zerolinecolor: '#2f334d', showbackground: false,
                tickfont: { color: '#565f89' }, titlefont: { color: '#94a3b8' }
            },
            yaxis: {
                title: 'Component 2',
                gridcolor: '#2f334d', zerolinecolor: '#2f334d', showbackground: false,
                tickfont: { color: '#565f89' }, titlefont: { color: '#94a3b8' }
            },
            zaxis: {
                title: 'Component 3',
                gridcolor: '#2f334d', zerolinecolor: '#2f334d', showbackground: false,
                tickfont: { color: '#565f89' }, titlefont: { color: '#94a3b8' }
            },
            camera: { eye: { x: 1.6, y: 1.2, z: 1.2 } }
        },
        showlegend: true,
        legend: { font: { color: '#c0caf5' }, bgcolor: 'rgba(0,0,0,0)' }
    };

    Plotly.newPlot('latent3dChart', [traceNormal, traceDrift, traceTarget], layout, { displayModeBar: false });
}

// --- 7. PDF REPORT GENERATION ---
async function downloadReport() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    let currentY = 15; // Start Y position

    // Helper to check page bounds
    function checkPageBreak(heightNeeded) {
        if (currentY + heightNeeded > 280) { // A4 height is ~297mm
            doc.addPage();
            currentY = 20; // Reset to top margin
        }
    }

    // 1. Header & Branding
    doc.setFillColor(30, 41, 59); // Dark blue
    doc.rect(0, 0, 210, 40, 'F');

    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor(255, 255, 255);
    doc.text("DriftSense Audit Report", 15, 25);

    doc.setFontSize(10);
    doc.setTextColor(148, 163, 184); // Slate-400
    const dateStr = new Date().toLocaleString();
    doc.text(`Generated: ${dateStr}`, 15, 35);

    currentY = 55; // Move below header

    // 2. Statistics Section
    doc.setTextColor(30, 41, 59); // Slate-800
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text(`Trace Index: ${currentDriftIndex}`, 15, currentY);
    currentY += 10;

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");

    if (globalData) {
        doc.text(`Reconstruction Error: ${Number(globalData.reconstruction_errors[currentDriftIndex]).toFixed(5)}`, 15, currentY);
        currentY += 6;
        doc.text(`Threshold: ${Number(globalData.threshold).toFixed(5)}`, 15, currentY);
        currentY += 6;
        doc.text(`Model Mode: ${globalData.mode || 'Unsupervised'}`, 15, currentY);
        currentY += 15;
    }

    // 3. AI Insight Section
    checkPageBreak(40); // Ensure space for title + some text
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(37, 99, 235); // Blue-600
    doc.text("AI Drift Insight:", 15, currentY);
    currentY += 8;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(30, 41, 59);

    let aiText = document.getElementById('geminiResponse').innerText || "AI analysis pending...";
    aiText = aiText.replace(/Gemini is thinking\.\.\./g, "Analysis pending generation.");

    const splitText = doc.splitTextToSize(aiText, 180);
    doc.text(splitText, 15, currentY);
    currentY += (splitText.length * 5) + 15;

    // 4. Screenshots of Charts
    try {
        const lineChart = document.getElementById('modalChart');
        const latentChart = document.getElementById('latent3dChart');
        const cfgContainer = document.getElementById('cfgNetwork');

        // A. Reconstruction Error Chart (Canvas)
        if (lineChart) {
            checkPageBreak(70);
            doc.setFont("helvetica", "bold");
            doc.setTextColor(37, 99, 235);
            doc.text("Local Reconstruction Error:", 15, currentY);
            currentY += 5;

            const canvasImg = lineChart.toDataURL("image/png");
            doc.addImage(canvasImg, 'PNG', 15, currentY, 180, 60);
            currentY += 70;
        }

        // B. 3D Latent Chart (Plotly)
        if (latentChart) {
            checkPageBreak(90);
            doc.setFont("helvetica", "bold");
            doc.setTextColor(37, 99, 235);
            doc.text("Latent Space Visualization:", 15, currentY);
            currentY += 5;

            // Plotly export
            await Plotly.toImage(latentChart, { format: 'png', width: 800, height: 400 })
                .then(function (dataUrl) {
                    doc.addImage(dataUrl, 'PNG', 15, currentY, 180, 90);
                });
            currentY += 95;
        }

        // C. Control Flow Graph (HTML DOM)
        if (cfgContainer) {
            checkPageBreak(100);
            doc.setFont("helvetica", "bold");
            doc.setTextColor(37, 99, 235);
            doc.text("Control Flow Graph Context:", 15, currentY);
            currentY += 5;

            // Use html2canvas to capture the div
            await html2canvas(cfgContainer, {
                backgroundColor: '#1a1b26', // Keep dark theme background
                scale: 2 // High res
            }).then(canvas => {
                const imgData = canvas.toDataURL('image/png');
                doc.addImage(imgData, 'PNG', 15, currentY, 180, 80);
            });
            currentY += 90;
        }

    } catch (err) {
        console.error("Chart capture failed", err);
        doc.setTextColor(220, 38, 38);
        doc.text(`Error generating charts: ${err.message}`, 15, currentY);
    }

    // SAVE
    doc.save(`DriftReport_Trace_${currentDriftIndex}.pdf`);
}