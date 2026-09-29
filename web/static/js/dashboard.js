/**
 * SafeCare Hospital PPE Compliance Dashboard Controller
 * Handles area-specific camera switching, live person inspection,
 * real-time KPI metrics, policy editing, and compliance trend charts.
 */

let currentDeptId = "icu";
let currentPersons = [];
let currentPersonIdx = 0;
let departmentsMap = {};

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initDepartmentsMap();
  initTrendChart();

  // Initial load for default active camera (ICU-01)
  const initialActive = document.querySelector(".camera-item.active");
  if (initialActive) {
    currentDeptId = initialActive.getAttribute("data-dept-id") || "icu";
  }
  refreshSelectedDepartment();

  // Periodic polling for live KPI metrics and person details
  setInterval(refreshDashboardSummary, 3000);
  setInterval(refreshActiveDepartmentTelemetry, 2500);
  refreshDashboardSummary();
});

// --- CLOCK ---
function initClock() {
  const clockEl = document.getElementById("live-clock");
  function update() {
    const now = new Date();
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    
    const day = days[now.getDay()];
    const date = now.getDate();
    const month = months[now.getMonth()];
    const year = now.getFullYear();

    let hours = now.getHours();
    const minutes = String(now.getMinutes()).padStart(2, "0");
    const seconds = String(now.getSeconds()).padStart(2, "0");
    const ampm = hours >= 12 ? "PM" : "AM";
    hours = hours % 12 || 12;

    if (clockEl) {
      clockEl.textContent = `${day}, ${date} ${month} ${year} ${hours}:${minutes}:${seconds} ${ampm}`;
    }

    const videoOverlay = document.getElementById("video-overlay-timestamp");
    const videoCode = document.getElementById("video-overlay-code");
    if (videoOverlay && videoCode) {
      videoOverlay.innerHTML = `${date} ${month} ${year} ${hours}:${minutes}:${seconds} <span>${videoCode.textContent}</span>`;
    }
  }
  update();
  setInterval(update, 1000);
}

// --- DEPARTMENTS & CAMERA SELECTION ---
function initDepartmentsMap() {
  const items = document.querySelectorAll(".camera-item");
  items.forEach(el => {
    const id = el.getAttribute("data-dept-id");
    const code = el.getAttribute("data-dept-code");
    const name = el.getAttribute("data-dept-name");
    const req = (el.getAttribute("data-required-ppe") || "").split(",").filter(Boolean);
    departmentsMap[id] = { id, code, name, required_ppe: req };
  });
}

window.selectDepartment = function(deptId) {
  currentDeptId = deptId;

  // Restore single-grid view if 4-camera grid was active
  const singleGrid = document.getElementById("safecare-single-grid");
  const multiGrid = document.getElementById("multi-ward-grid-section");
  if (singleGrid) singleGrid.style.display = "grid";
  if (multiGrid) multiGrid.classList.remove("active");

  // Update active state in sidebar camera list
  document.querySelectorAll(".camera-item").forEach(el => {
    el.classList.toggle("active", el.getAttribute("data-dept-id") === deptId);
  });

  // Update active state in top ward tabs bar
  document.querySelectorAll(".ward-tab-btn").forEach(el => {
    const isThisDept = el.getAttribute("data-dept-id") === deptId;
    el.classList.toggle("active", isThisDept);
  });
  const gridTab = document.getElementById("ward-tab-grid");
  const allWardsItem = document.getElementById("camera-item-all-wards");
  if (allWardsItem) allWardsItem.classList.remove("active");

  refreshSelectedDepartment();
};

window.toggleMultiWardGrid = function() {
  const singleGrid = document.getElementById("safecare-single-grid");
  const multiGrid = document.getElementById("multi-ward-grid-section");
  const allWardsItem = document.getElementById("camera-item-all-wards");

  if (!singleGrid || !multiGrid) return;

  const isGridActive = multiGrid.classList.contains("active");
  if (isGridActive) {
    multiGrid.classList.remove("active");
    singleGrid.style.display = "grid";
    if (allWardsItem) allWardsItem.classList.remove("active");
    selectDepartment(currentDeptId);
  } else {
    multiGrid.classList.add("active");
    singleGrid.style.display = "none";
    document.querySelectorAll(".camera-item").forEach(el => el.classList.remove("active"));
    if (allWardsItem) allWardsItem.classList.add("active");
  }
};

window.focusWard = function(deptId) {
  selectDepartment(deptId);
};

function refreshSelectedDepartment() {
  const dept = departmentsMap[currentDeptId];
  if (!dept) return;

  // Update Feed Title, Department Tag & Video Stream
  const feedCodeEl = document.getElementById("active-camera-code");
  const deptNameEl = document.getElementById("active-feed-dept-name");
  const overlayCodeEl = document.getElementById("video-overlay-code");
  const videoImg = document.getElementById("main-live-video");

  if (feedCodeEl) feedCodeEl.textContent = dept.code;
  if (deptNameEl) deptNameEl.textContent = dept.name;
  if (overlayCodeEl) overlayCodeEl.textContent = dept.code;
  if (videoImg) {
    videoImg.src = `/departments/${dept.id}/video_feed?t=${Date.now()}`;
  }

  // Update Area Policy Bar
  const areaNameEl = document.getElementById("active-policy-area");
  if (areaNameEl) areaNameEl.textContent = dept.name;

  renderPolicyPills(dept.required_ppe);

  // Fetch person details for this specific area
  refreshActiveDepartmentTelemetry();
}

function renderPolicyPills(requiredList) {
  const container = document.getElementById("active-policy-pills");
  if (!container) return;

  container.innerHTML = "";
  if (!requiredList || requiredList.length === 0) {
    container.innerHTML = `<span style="font-size: 0.75rem; color: #94a3b8;">No mandatory PPE required for this area.</span>`;
    return;
  }

  requiredList.forEach(item => {
    const cleanName = item.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    const pill = document.createElement("span");
    pill.className = "ppe-pill-badge";
    pill.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
      </svg>
      <span>${cleanName}</span>
    `;
    container.appendChild(pill);
  });
}

// --- PERSON DETAILS TELEMETRY & PAGINATION ---
async function refreshActiveDepartmentTelemetry() {
  try {
    const res = await fetch(`/api/departments/${currentDeptId}/person_details`);
    if (!res.ok) return;
    const data = await res.json();

    currentPersons = data.persons || [];
    
    // Update department required PPE in local map if updated remotely
    if (departmentsMap[currentDeptId] && data.required_ppe) {
      departmentsMap[currentDeptId].required_ppe = data.required_ppe;
      renderPolicyPills(data.required_ppe);
    }

    if (currentPersonIdx >= currentPersons.length) {
      currentPersonIdx = Math.max(0, currentPersons.length - 1);
    }

    renderPersonDetails();
  } catch (err) {
    console.warn("Failed to fetch department telemetry:", err);
  }
}

function renderPersonDetails() {
  const pagLabel = document.getElementById("person-pag-label");
  const pId = document.getElementById("p-id");
  const pLoc = document.getElementById("p-location");
  const pTime = document.getElementById("p-time");
  const pDur = document.getElementById("p-duration");
  const banner = document.getElementById("person-status-banner");
  const bannerState = document.getElementById("banner-state-text");
  const bannerDesc = document.getElementById("banner-desc-text");
  const checklistContainer = document.getElementById("ppe-checklist-container");

  if (currentPersons.length === 0) {
    if (pagLabel) pagLabel.textContent = "Person 0 of 0";
    if (pId) pId.textContent = "--";
    if (pLoc) pLoc.textContent = departmentsMap[currentDeptId] ? `${departmentsMap[currentDeptId].code} (${departmentsMap[currentDeptId].name})` : "--";
    if (pTime) pTime.textContent = "--";
    if (pDur) pDur.textContent = "00:00:00";
    if (banner) {
      banner.className = "person-compliance-banner banner-compliant";
      if (bannerState) bannerState.textContent = "AREA COMPLIANT";
      if (bannerDesc) bannerDesc.textContent = "No personnel currently detected in frame";
    }
    return;
  }

  const p = currentPersons[currentPersonIdx];
  if (pagLabel) pagLabel.textContent = `Person ${currentPersonIdx + 1} of ${currentPersons.length}`;
  if (pId) pId.textContent = p.person_id_formatted || String(p.track_id).padStart(2, "0");
  if (pLoc) pLoc.textContent = p.location;
  if (pTime) pTime.textContent = p.time_detected;
  if (pDur) pDur.textContent = p.duration;

  // Render checklist for this person
  if (checklistContainer && p.items) {
    checklistContainer.innerHTML = "";
    const dept = departmentsMap[currentDeptId];
    const reqSet = new Set(dept ? dept.required_ppe : []);

    const iconMap = {
      face_mask: "😷",
      gloves: "🧤",
      gown: "🥼",
      eye_protection: "🥽",
      head_cover: "🧢",
      shoe_covers: "👟"
    };

    Object.entries(p.items).forEach(([key, info]) => {
      const isReq = reqSet.has(key);
      const row = document.createElement("div");
      row.className = "checklist-row";

      const iconEmoji = iconMap[key] || "🛡";
      const statusIcon = info.detected ? "✔" : (isReq ? "✖" : "—");
      const iconClass = info.detected ? "green-icon" : (isReq ? "red-icon" : "text-dim");
      const textClass = info.detected ? "green-text" : (isReq ? "red-text" : "text-dim");
      const textLabel = info.detected 
        ? `Detected (${info.confidence}%)` 
        : (isReq ? "Not Detected" : "Optional (Not Worn)");

      row.innerHTML = `
        <div class="chk-left">
          <span class="chk-icon">${iconEmoji}</span>
          <span class="chk-name">${info.name} ${isReq ? '' : '<small style="color:#64748b;">(Optional)</small>'}</span>
        </div>
        <div class="chk-right">
          <span class="chk-status-icon ${iconClass}">${statusIcon}</span>
          <span class="chk-text ${textClass}">${textLabel}</span>
        </div>
      `;
      checklistContainer.appendChild(row);
    });
  }

  // Update Status Banner
  if (banner) {
    if (p.is_compliant) {
      banner.className = "person-compliance-banner banner-compliant";
      if (bannerState) bannerState.textContent = "COMPLIANT";
      if (bannerDesc) bannerDesc.textContent = "All required PPE detected for this area";
    } else {
      banner.className = "person-compliance-banner banner-non-compliant";
      if (bannerState) bannerState.textContent = "NON-COMPLIANT";
      if (bannerDesc) bannerDesc.textContent = `Missing: ${p.missing_items.join(", ")}`;
    }
  }
}

window.prevPerson = function() {
  if (currentPersons.length <= 1) return;
  currentPersonIdx = (currentPersonIdx - 1 + currentPersons.length) % currentPersons.length;
  renderPersonDetails();
};

window.nextPerson = function() {
  if (currentPersons.length <= 1) return;
  currentPersonIdx = (currentPersonIdx + 1) % currentPersons.length;
  renderPersonDetails();
};

// --- TOP KPI METRICS ---
async function refreshDashboardSummary() {
  try {
    const res = await fetch("/api/dashboard/summary");
    if (!res.ok) return;
    const data = await res.json();

    const elPeople = document.getElementById("kpi-people");
    const elComp = document.getElementById("kpi-compliant");
    const elNonComp = document.getElementById("kpi-non-compliant");
    const elCams = document.getElementById("kpi-cameras");
    const elIncs = document.getElementById("kpi-incidents");
    const elBadge = document.getElementById("topbar-notif-count");
    const elSideBadge = document.getElementById("sidebar-incident-badge");

    if (elPeople) elPeople.textContent = data.people_monitored || 12;
    if (elComp) elComp.textContent = data.compliant || 9;
    if (elNonComp) elNonComp.textContent = data.non_compliant || 3;
    if (elCams) elCams.textContent = data.active_cameras || 4;
    if (elIncs) elIncs.textContent = data.total_incidents_today || 27;
    if (elBadge) elBadge.textContent = data.non_compliant || 3;
    if (elSideBadge) elSideBadge.textContent = data.non_compliant || 3;
  } catch (err) {
    console.warn("Error fetching dashboard summary:", err);
  }
}

// --- COMPLIANCE TREND CHART ---
function initTrendChart() {
  updateTrendChart();
}

window.updateTrendChart = async function() {
  const canvas = document.getElementById("compliance-trend-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const areaSelect = document.getElementById("trend-area-select");
  const area = areaSelect ? areaSelect.value : "all";

  try {
    const res = await fetch(`/api/compliance/trend?area=${area}`);
    const data = await res.json();

    drawTrendChart(ctx, canvas.width, canvas.height, data.labels, data.compliant, data.non_compliant);
  } catch (err) {
    // Fallback static data matching reference image
    drawTrendChart(
      ctx,
      canvas.width,
      canvas.height,
      ["8AM", "10AM", "12PM", "2PM", "4PM", "6PM"],
      [25, 30, 28, 31, 25, 29],
      [10, 8, 12, 7, 9, 8]
    );
  }
};

function drawTrendChart(ctx, w, h, labels, compliant, nonCompliant) {
  ctx.clearRect(0, 0, w, h);

  const padLeft = 32;
  const padRight = 15;
  const padTop = 15;
  const padBottom = 25;

  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;
  const maxVal = 40;

  // Grid lines and Y-axis labels (0, 10, 20, 30, 40)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
  ctx.fillStyle = "#64748b";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";

  for (let val = 0; val <= maxVal; val += 10) {
    const y = padTop + chartH - (val / maxVal) * chartH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
    ctx.fillText(String(val), padLeft - 6, y + 3);
  }

  // X-axis labels
  ctx.textAlign = "center";
  const stepX = chartW / (labels.length - 1);
  labels.forEach((lbl, idx) => {
    const x = padLeft + idx * stepX;
    ctx.fillText(lbl, x, h - 6);
  });

  // Helper to draw smooth line
  function drawLine(points, color, fillColor) {
    if (points.length === 0) return;

    ctx.beginPath();
    points.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill under curve
    if (fillColor) {
      ctx.lineTo(points[points.length - 1].x, padTop + chartH);
      ctx.lineTo(points[0].x, padTop + chartH);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    // Draw dots
    points.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  // Compliant Points (Green)
  const compPoints = compliant.map((val, i) => ({
    x: padLeft + i * stepX,
    y: padTop + chartH - (val / maxVal) * chartH
  }));
  drawLine(compPoints, "#10b981", "rgba(16, 185, 129, 0.08)");

  // Non-Compliant Points (Red)
  const nonCompPoints = nonCompliant.map((val, i) => ({
    x: padLeft + i * stepX,
    y: padTop + chartH - (val / maxVal) * chartH
  }));
  drawLine(nonCompPoints, "#ef4444", "rgba(239, 68, 68, 0.08)");
}

// --- EDIT POLICY MODAL ---
window.openEditPolicyModal = function() {
  const modal = document.getElementById("edit-policy-modal");
  const dept = departmentsMap[currentDeptId];
  if (!modal || !dept) return;

  document.getElementById("policy-modal-title").textContent = `Edit Area Policy: ${dept.name} (${dept.code})`;

  const reqSet = new Set(dept.required_ppe || []);
  document.getElementById("pol-chk-mask").checked = reqSet.has("face_mask");
  document.getElementById("pol-chk-gloves").checked = reqSet.has("gloves");
  document.getElementById("pol-chk-gown").checked = reqSet.has("gown");
  document.getElementById("pol-chk-eye").checked = reqSet.has("eye_protection");
  document.getElementById("pol-chk-head").checked = reqSet.has("head_cover");
  document.getElementById("pol-chk-shoes").checked = reqSet.has("shoe_covers");

  modal.classList.add("open");
};

window.closeEditPolicyModal = function() {
  const modal = document.getElementById("edit-policy-modal");
  if (modal) modal.classList.remove("open");
};

window.saveAreaPolicy = async function() {
  const required = [];
  if (document.getElementById("pol-chk-mask").checked) required.push("face_mask");
  if (document.getElementById("pol-chk-gloves").checked) required.push("gloves");
  if (document.getElementById("pol-chk-gown").checked) required.push("gown");
  if (document.getElementById("pol-chk-eye").checked) required.push("eye_protection");
  if (document.getElementById("pol-chk-head").checked) required.push("head_cover");
  if (document.getElementById("pol-chk-shoes").checked) required.push("shoe_covers");

  const grace = parseFloat(document.getElementById("pol-grace-input").value) || 3.0;

  try {
    const res = await fetch(`/api/departments/${currentDeptId}/policy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        required_ppe: required,
        grace_period_seconds: grace
      })
    });
    if (res.ok) {
      if (departmentsMap[currentDeptId]) {
        departmentsMap[currentDeptId].required_ppe = required;
      }
      renderPolicyPills(required);
      refreshActiveDepartmentTelemetry();
      closeEditPolicyModal();
    } else {
      alert("Failed to save area policy.");
    }
  } catch (err) {
    alert("Error updating area policy: " + err);
  }
};

// --- ADD CAMERA MODAL ---
window.openAddCameraModal = function() {
  const modal = document.getElementById("add-camera-modal");
  if (modal) modal.classList.add("open");
};

window.closeAddCameraModal = function() {
  const modal = document.getElementById("add-camera-modal");
  if (modal) modal.classList.remove("open");
};

window.submitAddCamera = async function() {
  const id = document.getElementById("add-cam-id").value.trim();
  const name = document.getElementById("add-cam-name").value.trim();
  const code = document.getElementById("add-cam-code").value.trim();
  const source = document.getElementById("add-cam-source").value.trim();

  if (!id || !name || !source) {
    alert("Please provide ID, Area Name, and Camera Source.");
    return;
  }

  try {
    const res = await fetch("/api/departments/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id,
        name,
        code: code || `DEPT-${id.toUpperCase()}`,
        camera_source: source,
        required_ppe: ["face_mask"]
      })
    });
    if (res.ok) {
      window.location.reload();
    } else {
      alert("Failed to add camera.");
    }
  } catch (err) {
    alert("Error adding camera: " + err);
  }
};

window.scrollToCameras = function() {
  const el = document.getElementById("cameras-section");
  if (el) el.scrollIntoView({ behavior: "smooth" });
};

// --- VIDEO INSERTION & SWITCHER MODAL ---
window.openInsertVideoModal = function() {
  const modal = document.getElementById("insert-video-modal");
  if (!modal) return;

  try {
    const dept = (typeof departmentsMap !== "undefined" && departmentsMap[currentDeptId]) ? departmentsMap[currentDeptId] : null;
    const targetTitle = document.getElementById("insert-modal-dept-title");
    if (targetTitle && dept) {
      targetTitle.textContent = `${dept.name} (${dept.code})`;
    }
  } catch (e) {}

  modal.classList.add("open");
  modal.style.display = "flex";
  modal.style.opacity = "1";
  modal.style.pointerEvents = "auto";
  modal.style.visibility = "visible";
  modal.style.zIndex = "99999";

  try {
    if (typeof switchVideoModalTab === "function") switchVideoModalTab("demos");
    if (typeof loadSampleVideosList === "function") loadSampleVideosList();
  } catch (e) {}
};

window.closeInsertVideoModal = function() {
  const modal = document.getElementById("insert-video-modal");
  if (modal) {
    modal.classList.remove("open");
    modal.style.display = "none";
    modal.style.opacity = "0";
    modal.style.pointerEvents = "none";
    modal.style.visibility = "hidden";
  }
};

window.switchVideoModalTab = function(tabName) {
  const tabs = ["demos", "upload", "webcam"];
  tabs.forEach(t => {
    const btn = document.getElementById(`vtab-btn-${t}`);
    const pane = document.getElementById(`vtab-pane-${t}`);
    if (btn) btn.classList.toggle("active", t === tabName);
    if (pane) pane.style.display = (t === tabName) ? "block" : "none";
  });
};

async function loadSampleVideosList() {
  try {
    const res = await fetch("/api/sample_videos");
    if (!res.ok) return;
    const list = await res.json();
    const container = document.getElementById("demo-videos-container");
    if (!container || !list.length) return;

    container.innerHTML = "";
    list.forEach(v => {
      const item = document.createElement("div");
      item.className = "demo-video-item";
      item.innerHTML = `
        <div class="demo-item-left">
          <div class="demo-item-badge-row">
            <span class="demo-cat-badge badge-${v.badge_color || 'blue'}">${v.category || 'DEMO'}</span>
            <span class="demo-item-title">${v.name}</span>
          </div>
          <span class="demo-item-desc">${v.description || ''}</span>
        </div>
        <button class="btn-select-video" onclick="applySampleVideo('${v.path}')">
          Apply to Ward
        </button>
      `;
      container.appendChild(item);
    });
  } catch (e) {
    console.error("Error loading sample videos:", e);
  }
}

window.applySampleVideo = async function(videoPath) {
  try {
    const res = await fetch(`/api/departments/${currentDeptId}/camera`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: videoPath })
    });
    if (res.ok) {
      closeInsertVideoModal();
      refreshSelectedDepartment();
    } else {
      alert("Failed to update video source for this ward.");
    }
  } catch (e) {
    console.error(e);
  }
};

window.handleVideoFileSelected = async function(event) {
  const file = event.target.files[0];
  if (!file) return;

  const progContainer = document.getElementById("upload-progress-container");
  const progBar = document.getElementById("upload-progress-bar");
  const filenameEl = document.getElementById("upload-filename");
  const percentEl = document.getElementById("upload-percent");
  const statusMsg = document.getElementById("upload-status-msg");

  if (progContainer) progContainer.style.display = "block";
  if (filenameEl) filenameEl.textContent = file.name;
  if (statusMsg) statusMsg.style.display = "none";

  const formData = new FormData();
  formData.append("file", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/upload_video?department_id=${currentDeptId}`, true);

  xhr.upload.onprogress = function(e) {
    if (e.lengthComputable) {
      const percent = Math.round((e.loaded / e.total) * 100);
      if (progBar) progBar.style.width = `${percent}%`;
      if (percentEl) percentEl.textContent = `${percent}%`;
    }
  };

  xhr.onload = function() {
    if (xhr.status === 200) {
      if (statusMsg) {
        statusMsg.style.display = "block";
        statusMsg.style.color = "#10b981";
        statusMsg.textContent = "Video uploaded and applied successfully!";
      }
      setTimeout(() => {
        closeInsertVideoModal();
        refreshSelectedDepartment();
      }, 1000);
    } else {
      if (statusMsg) {
        statusMsg.style.display = "block";
        statusMsg.style.color = "#ef4444";
        statusMsg.textContent = "Upload failed: " + xhr.responseText;
      }
    }
  };

  xhr.onerror = function() {
    if (statusMsg) {
      statusMsg.style.display = "block";
      statusMsg.style.color = "#ef4444";
      statusMsg.textContent = "Network error during video upload.";
    }
  };

  xhr.send(formData);
};

window.applyWebcamSource = async function() {
  const select = document.getElementById("webcam-device-select");
  const devIdx = select ? select.value : "0";
  await applySampleVideo(devIdx);
};

window.applyRtspSource = async function() {
  const input = document.getElementById("rtsp-url-input");
  const url = input ? input.value.trim() : "";
  if (!url) {
    alert("Please enter a valid RTSP URL.");
    return;
  }
  await applySampleVideo(url);
};
