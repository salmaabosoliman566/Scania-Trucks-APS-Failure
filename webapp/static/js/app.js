/**
 * Scania Trucks APS Failure Prediction Web Application
 * Client-Side Application Logic (ES6 Vanilla JavaScript)
 */

// Application State
let appState = {
  presets: null,
  activeTab: 'homeView',
  history: [],
  batchResults: [],
  filteredBatchResults: []
};

// 20 Core Model Features
const TOP_FEATURES = [
  'bj_000', 'aq_000', 'ci_000', 'dn_000', 'ap_000', 
  'bb_000', 'bv_000', 'bu_000', 'cq_000', 'ck_000', 
  'bt_000', 'aa_000', 'bh_000', 'ah_000', 'cc_000', 
  'bg_000', 'an_000', 'bx_000', 'ao_000', 'cs_002'
];

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  setupDragAndDrop();
  await checkSystemHealth();
  await loadPresetsData();
  loadSessionHistory();
});

/* ==========================================================================
   NAVIGATION
   ========================================================================== */
function setupNavigation() {
  const navBtns = document.querySelectorAll('.nav-item-btn');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      switchTab(targetId);
    });
  });

  const mobileToggle = document.getElementById('mobileNavToggle');
  const navLinks = document.getElementById('navLinks');
  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener('click', () => {
      navLinks.classList.toggle('mobile-open');
    });
  }
}

function switchTab(tabId) {
  // Update views
  document.querySelectorAll('.view-section').forEach(view => {
    view.classList.remove('active');
  });
  const targetView = document.getElementById(tabId);
  if (targetView) {
    targetView.classList.add('active');
    appState.activeTab = tabId;
  }

  // Update nav buttons
  document.querySelectorAll('.nav-item-btn').forEach(btn => {
    btn.classList.remove('active');
    if (btn.getAttribute('data-target') === tabId) {
      btn.classList.add('active');
    }
  });

  // Close mobile nav if open
  const navLinks = document.getElementById('navLinks');
  if (navLinks) navLinks.classList.remove('mobile-open');

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ==========================================================================
   SYSTEM HEALTH & PRESETS LOADING
   ========================================================================== */
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const statusText = document.getElementById('modelStatusText');
    const statusPill = document.getElementById('systemStatusPill');

    if (data.model_loaded) {
      statusText.textContent = 'Model: Active (Tuned Random Forest)';
      statusPill.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      statusText.textContent = 'Model: Initializing Pipeline...';
      statusPill.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    }
  } catch (err) {
    console.warn('Backend API connection pending:', err);
  }
}

async function loadPresetsData() {
  try {
    const res = await fetch('/static/data/presets.json');
    if (res.ok) {
      const data = await res.json();
      appState.presets = data.presets;
      // Pre-fill form with healthy preset by default
      loadPreset('healthy', false);
    }
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

/* ==========================================================================
   PRESET SELECTION
   ========================================================================== */
function loadPreset(presetKey, autoRun = true) {
  // Update active preset button style
  document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
  const btnMap = {
    'healthy': 'btnPresetHealthy',
    'critical': 'btnPresetCritical',
    'borderline': 'btnPresetBorderline',
    'missing_sensors': 'btnPresetMissing'
  };
  const activeBtn = document.getElementById(btnMap[presetKey]);
  if (activeBtn) activeBtn.classList.add('active');

  if (!appState.presets || !appState.presets[presetKey]) return;

  const preset = appState.presets[presetKey];
  const features = preset.features;

  TOP_FEATURES.forEach(feat => {
    const input = document.getElementById(`input_${feat}`);
    if (input) {
      input.value = (features[feat] !== null && features[feat] !== undefined) ? features[feat] : '';
    }
  });

  showToast(`Loaded profile: ${preset.title}`, 'info');

  if (autoRun) {
    // Automatically trigger prediction for immediate visual feedback
    document.getElementById('singlePredictionForm').requestSubmit();
  }
}

function resetForm() {
  document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.sensor-input').forEach(input => input.value = '');
  document.getElementById('predictionResultView').classList.remove('active');
  document.getElementById('resultPlaceholder').style.display = 'block';
  showToast('Sensor inputs cleared', 'info');
}

/* ==========================================================================
   SINGLE TRUCK PREDICTION
   ========================================================================== */
async function handleSinglePredict(event) {
  event.preventDefault();

  const submitBtn = document.getElementById('btnRunPrediction');
  const btnText = document.getElementById('predictBtnText');
  const spinner = document.getElementById('predictSpinner');

  // Collect features from inputs
  const payloadFeatures = {};
  TOP_FEATURES.forEach(feat => {
    const input = document.getElementById(`input_${feat}`);
    if (input && input.value.trim() !== '') {
      payloadFeatures[feat] = parseFloat(input.value.trim());
    } else {
      payloadFeatures[feat] = null; // Passed as null for median imputation
    }
  });

  // UI Loading State
  submitBtn.disabled = true;
  btnText.textContent = 'Evaluating ML Pipeline...';
  spinner.style.display = 'inline-block';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: payloadFeatures })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Prediction failed');
    }

    const result = await response.json();
    renderPredictionResult(result);
    addToHistory(result, payloadFeatures);
    showToast(`Inference Complete: ${result.status_headline}`, result.prediction === 1 ? 'error' : 'success');

  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    submitBtn.disabled = false;
    btnText.textContent = 'Run Machine Learning Inference';
    spinner.style.display = 'none';
  }
}

function renderPredictionResult(result) {
  const placeholder = document.getElementById('resultPlaceholder');
  const resultView = document.getElementById('predictionResultView');
  placeholder.style.display = 'none';
  resultView.classList.add('active');

  // Status Badge
  const badge = document.getElementById('resultStatusBadge');
  const badgeText = document.getElementById('resultBadgeText');
  const badgeIcon = document.getElementById('resultBadgeIcon');

  badge.className = `status-badge ${result.badge_class}`;
  badgeText.textContent = result.status_headline;
  badgeIcon.textContent = result.prediction === 1 ? '▲' : '●';

  // Gauge Meter
  const probPercent = (result.failure_probability * 100).toFixed(1);
  const scoreText = document.getElementById('gaugeScoreText');
  scoreText.textContent = `${probPercent}%`;

  const gaugeArc = document.getElementById('gaugeArc');
  // Total arc circumference is 251.3
  const maxCircumference = 251.3;
  const strokeOffset = maxCircumference - (maxCircumference * result.failure_probability);
  gaugeArc.style.strokeDashoffset = strokeOffset;

  // Arc Color
  if (result.failure_probability >= 0.75) {
    gaugeArc.style.stroke = '#ef4444'; // Red
    scoreText.style.color = '#ef4444';
  } else if (result.failure_probability >= 0.50) {
    gaugeArc.style.stroke = '#f59e0b'; // Amber
    scoreText.style.color = '#f59e0b';
  } else if (result.failure_probability >= 0.20) {
    gaugeArc.style.stroke = '#3b82f6'; // Blue
    scoreText.style.color = '#3b82f6';
  } else {
    gaugeArc.style.stroke = '#10b981'; // Green
    scoreText.style.color = '#10b981';
  }

  // Recommendations and Financial Analysis
  document.getElementById('resultRecommendationText').textContent = result.recommendation;
  document.getElementById('resultFinancialTag').textContent = result.cost_saving_msg;

  // Anomaly Highlights
  const anomaliesList = document.getElementById('anomaliesList');
  if (result.top_anomalies && result.top_anomalies.length > 0) {
    anomaliesList.innerHTML = result.top_anomalies.map(a => `
      <div style="display:flex; justify-content:space-between; margin-bottom: 0.35rem; border-bottom: 1px solid rgba(255,255,255,0.04); padding-bottom: 0.2rem;">
        <span><strong>${a.feature}</strong> (${a.name})</span>
        <span style="font-family:monospace; color:${a.severity === 'high' ? '#f87171' : '#fbbf24'};">${Number(a.value).toLocaleString()} ${a.unit}</span>
      </div>
    `).join('');
  } else {
    anomaliesList.innerHTML = '<span style="color:var(--accent-emerald);">All 20 sensor telemetry values are within normal operating thresholds.</span>';
  }
}

/* ==========================================================================
   BATCH CSV TELEMETRY
   ========================================================================== */
function setupDragAndDrop() {
  const dropZone = document.getElementById('dropZone');
  if (!dropZone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleCsvFile(files[0]);
    }
  });
}

async function handleCsvFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.csv')) {
    showToast('Please select a valid CSV file (.csv)', 'error');
    return;
  }

  document.getElementById('uploadTitle').textContent = `Processing: ${file.name}`;
  document.getElementById('batchLoading').style.display = 'block';
  document.getElementById('batchResultsSection').style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/predict-batch', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Batch prediction failed');
    }

    const batchData = await response.json();
    appState.batchResults = batchData.results;
    appState.filteredBatchResults = [...batchData.results];

    renderBatchSummary(batchData);
    renderBatchTable(appState.filteredBatchResults);

    document.getElementById('batchResultsSection').style.display = 'block';
    showToast(`Processed ${batchData.total_trucks} trucks successfully!`, 'success');

  } catch (err) {
    showToast(`Batch Error: ${err.message}`, 'error');
  } finally {
    document.getElementById('batchLoading').style.display = 'none';
  }
}

function renderBatchSummary(data) {
  document.getElementById('batchTotalCount').textContent = data.total_trucks.toLocaleString();
  document.getElementById('batchFailuresCount').textContent = data.failures_detected.toLocaleString();
  document.getElementById('batchHealthyCount').textContent = data.healthy_trucks.toLocaleString();
  document.getElementById('batchSavingsVal').textContent = `$${data.estimated_cost_savings.toLocaleString()}`;
}

function renderBatchTable(rows) {
  const tbody = document.getElementById('batchTableBody');
  tbody.innerHTML = '';

  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 2rem; color: var(--text-muted);">No records matching current filter.</td></tr>`;
    return;
  }

  rows.forEach(r => {
    const tr = document.createElement('tr');
    const isFail = r.prediction === 1;
    const badgeColor = isFail ? 'rgba(217, 37, 52, 0.2)' : 'rgba(16, 185, 129, 0.2)';
    const textColor = isFail ? '#f87171' : '#34d399';

    tr.innerHTML = `
      <td><strong>${r.truck_id}</strong></td>
      <td>
        <span style="display:inline-block; padding: 2px 8px; border-radius: 4px; font-weight:700; background:${badgeColor}; color:${textColor};">
          ${r.status}
        </span>
      </td>
      <td style="font-family: monospace; font-weight: 600;">${(r.failure_probability * 100).toFixed(2)}%</td>
      <td><span style="font-size:0.75rem; font-weight:700; color:${textColor};">${r.risk_badge}</span></td>
      <td style="color: var(--text-secondary);">${r.recommended_action}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterBatchTable() {
  const query = document.getElementById('tableSearchInput').value.toLowerCase();
  appState.filteredBatchResults = appState.batchResults.filter(r => r.truck_id.toLowerCase().includes(query));
  renderBatchTable(appState.filteredBatchResults);
}

function filterByStatus(type) {
  const query = document.getElementById('tableSearchInput').value.toLowerCase();
  let base = appState.batchResults.filter(r => r.truck_id.toLowerCase().includes(query));
  
  if (type === 'failure') {
    appState.filteredBatchResults = base.filter(r => r.prediction === 1);
  } else if (type === 'healthy') {
    appState.filteredBatchResults = base.filter(r => r.prediction === 0);
  } else {
    appState.filteredBatchResults = base;
  }
  renderBatchTable(appState.filteredBatchResults);
}

function exportBatchCsv() {
  if (!appState.filteredBatchResults || appState.filteredBatchResults.length === 0) {
    showToast('No records to export', 'error');
    return;
  }

  const headers = ['truck_id', 'prediction', 'status', 'failure_probability', 'risk_badge', 'recommended_action'];
  const csvRows = [headers.join(',')];

  appState.filteredBatchResults.forEach(r => {
    csvRows.push([
      `"${r.truck_id}"`,
      r.prediction,
      `"${r.status}"`,
      r.failure_probability,
      `"${r.risk_badge}"`,
      `"${r.recommended_action}"`
    ].join(','));
  });

  const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.setAttribute('href', url);
  a.setAttribute('download', `scania_aps_predictions_${new Date().toISOString().slice(0,10)}.csv`);
  a.click();
  showToast('Exported predictions to CSV', 'success');
}

/* ==========================================================================
   SESSION HISTORY
   ========================================================================== */
function addToHistory(result, features) {
  const item = {
    id: `HIST-${Date.now()}`,
    timestamp: new Date().toLocaleTimeString(),
    result: result,
    features: features
  };

  appState.history.unshift(item);
  if (appState.history.length > 20) appState.history.pop();

  saveSessionHistory();
  renderHistoryTable();
}

function renderHistoryTable() {
  const tbody = document.getElementById('historyTableBody');
  const counter = document.getElementById('historyCounter');
  if (counter) counter.textContent = appState.history.length;

  if (!tbody) return;

  if (appState.history.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; padding: 2.5rem; color: var(--text-muted);">
          No diagnostics run yet. Use the Single Diagnostic tab to generate your first prediction.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = '';
  appState.history.forEach((h, idx) => {
    const isFail = h.result.prediction === 1;
    const textColor = isFail ? '#f87171' : '#34d399';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${h.timestamp}</td>
      <td><span style="font-weight:700; color:${textColor};">${h.result.status}</span></td>
      <td style="font-family:monospace;">${(h.result.failure_probability * 100).toFixed(1)}%</td>
      <td><span style="font-size:0.75rem; font-weight:700; color:${textColor};">${h.result.risk_tier}</span></td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="reloadHistoryItem(${idx})">Reload Telemetry</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function reloadHistoryItem(index) {
  const item = appState.history[index];
  if (!item) return;

  TOP_FEATURES.forEach(feat => {
    const input = document.getElementById(`input_${feat}`);
    if (input) {
      input.value = item.features[feat] !== null ? item.features[feat] : '';
    }
  });

  switchTab('predictView');
  renderPredictionResult(item.result);
  showToast(`Reloaded diagnostic from ${item.timestamp}`, 'info');
}

function clearHistory() {
  appState.history = [];
  saveSessionHistory();
  renderHistoryTable();
  showToast('Session history cleared', 'info');
}

function saveSessionHistory() {
  try {
    sessionStorage.setItem('scania_aps_history', JSON.stringify(appState.history));
  } catch (e) {}
}

function loadSessionHistory() {
  try {
    const saved = sessionStorage.getItem('scania_aps_history');
    if (saved) {
      appState.history = JSON.parse(saved);
      renderHistoryTable();
    }
  } catch (e) {}
}

/* ==========================================================================
   TOAST NOTIFICATIONS
   ========================================================================== */
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const iconMap = {
    'success': '✓',
    'error': '⚠',
    'info': 'ℹ'
  };

  toast.innerHTML = `
    <span style="font-weight:900;">${iconMap[type] || 'ℹ'}</span>
    <span>${message}</span>
  `;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(16px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
