// Interactive Zone & Polygon Policy Editor
document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("zone-canvas");
  const ctx = canvas.getContext("2d");
  const zonesList = document.getElementById("zones-list");
  const refreshFrameBtn = document.getElementById("refresh-frame-btn");
  const clearPolygonBtn = document.getElementById("clear-polygon-btn");
  const saveZoneBtn = document.getElementById("save-zone-btn");
  const deleteZoneBtn = document.getElementById("delete-zone-btn");

  const zoneNameInput = document.getElementById("zone-name");
  const zoneCodeInput = document.getElementById("zone-code");
  const zoneRiskSelect = document.getElementById("zone-risk");
  const zoneGraceInput = document.getElementById("zone-grace");
  const zoneSeveritySelect = document.getElementById("zone-severity");
  const zoneColorPicker = document.getElementById("zone-color-picker");

  let backgroundImage = new Image();
  let zones = [];
  let currentPoints = []; // Normalized [0.0 - 1.0] coordinates
  let activeEditingZoneId = null;

  // Load snapshot frame from camera
  function loadBackgroundFrame() {
    backgroundImage.crossOrigin = "anonymous";
    backgroundImage.src = `/api/stream/snapshot?t=${Date.now()}`;
    backgroundImage.onload = () => {
      renderCanvas();
    };
  }

  // Fetch configured zones
  async function fetchZones() {
    try {
      const res = await fetch("/api/zones");
      if (!res.ok) return;
      zones = await res.json();
      renderZonesList();
      renderCanvas();
    } catch (err) {
      console.error("Failed to load zones:", err);
    }
  }

  function renderZonesList() {
    zonesList.innerHTML = zones.map(z => {
      const isSelected = z.id === activeEditingZoneId;
      const rgb = z.color;
      return `
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0.75rem; background: ${isSelected ? 'rgba(6, 182, 212, 0.15)' : 'var(--bg-primary)'}; border: 1px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--border-color)'}; border-radius: 6px; cursor: pointer;"
             onclick="selectZoneForEdit('${z.id}')">
          <div>
            <span class="zone-chip" style="background: rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.2); color: rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]}); border: 1px solid rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]});">
              ${z.code}
            </span>
            <span style="font-size: 0.82rem; font-weight: 600;">${z.name}</span>
          </div>
          <span style="font-size: 0.72rem; color: var(--text-muted);">${z.risk_level}</span>
        </div>
      `;
    }).join("");
  }

  // Draw background frame, saved zones, and currently drawing polygon
  function renderCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw video background
    if (backgroundImage.complete && backgroundImage.naturalWidth !== 0) {
      ctx.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#1e293b";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // Draw all saved zones
    zones.forEach(z => {
      if (z.polygon.length < 3) return;
      const rgb = z.color;
      const isEditing = z.id === activeEditingZoneId;

      ctx.beginPath();
      z.polygon.forEach((pt, idx) => {
        const px = pt[0] * canvas.width;
        const py = pt[1] * canvas.height;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();

      // Fill
      ctx.fillStyle = isEditing ? `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.45)` : `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.25)`;
      ctx.fill();

      // Border
      ctx.strokeStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
      ctx.lineWidth = isEditing ? 3 : 2;
      ctx.stroke();

      // Label
      const p0 = z.polygon[0];
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 13px -apple-system, sans-serif";
      ctx.fillText(`[${z.code}] ${z.name}`, p0[0] * canvas.width + 6, p0[1] * canvas.height + 18);
    });

    // Draw active drawing polygon points & lines
    if (currentPoints.length > 0) {
      ctx.beginPath();
      currentPoints.forEach((pt, idx) => {
        const px = pt[0] * canvas.width;
        const py = pt[1] * canvas.height;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);

        // Draw vertex point
        ctx.fillStyle = "#06b6d4";
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      if (currentPoints.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = "rgba(6, 182, 212, 0.25)";
        ctx.fill();
      }
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // Handle canvas clicks to draw polygon vertices
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Normalize to [0.0 - 1.0]
    const normX = Math.max(0, Math.min(1.0, x / canvas.width));
    const normY = Math.max(0, Math.min(1.0, y / canvas.height));

    currentPoints.push([normX, normY]);
    renderCanvas();
  });

  // Select a zone for editing
  window.selectZoneForEdit = (zoneId) => {
    const zone = zones.find(z => z.id === zoneId);
    if (!zone) return;

    activeEditingZoneId = zone.id;
    currentPoints = [...zone.polygon];

    zoneNameInput.value = zone.name;
    zoneCodeInput.value = zone.code;
    zoneRiskSelect.value = zone.risk_level;
    zoneGraceInput.value = zone.grace_period_seconds;
    zoneSeveritySelect.value = zone.alert_severity;

    // Convert RGB to Hex
    const hex = "#" + zone.color.map(c => c.toString(16).padStart(2, "0")).join("");
    zoneColorPicker.value = hex;

    // Check checkboxes
    document.querySelectorAll("input[name='req_ppe']").forEach(cb => {
      cb.checked = zone.required_ppe.includes(cb.value);
    });

    deleteZoneBtn.style.display = "inline-block";
    renderZonesList();
    renderCanvas();
  };

  // Clear current drawing
  clearPolygonBtn.addEventListener("click", () => {
    currentPoints = [];
    activeEditingZoneId = null;
    deleteZoneBtn.style.display = "none";
    document.getElementById("zone-form").reset();
    renderZonesList();
    renderCanvas();
  });

  refreshFrameBtn.addEventListener("click", loadBackgroundFrame);

  // Save Zone
  saveZoneBtn.addEventListener("click", async () => {
    if (currentPoints.length < 3) {
      alert("Please define a polygon with at least 3 points by clicking on the camera frame.");
      return;
    }

    if (!zoneNameInput.value.trim() || !zoneCodeInput.value.trim()) {
      alert("Please provide both Zone Name and Zone Code.");
      return;
    }

    // Parse Color Hex to RGB
    const hex = zoneColorPicker.value;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    // Get checked PPE
    const reqPPE = Array.from(document.querySelectorAll("input[name='req_ppe']:checked")).map(cb => cb.value);

    const zoneId = activeEditingZoneId || `zone_${zoneCodeInput.value.toLowerCase().replace(/[^a-z0-9]/g, "_")}`;

    const payload = {
      id: zoneId,
      name: zoneNameInput.value.trim(),
      code: zoneCodeInput.value.trim().toUpperCase(),
      risk_level: zoneRiskSelect.value,
      color: [r, g, b],
      polygon: currentPoints,
      required_ppe: reqPPE,
      optional_ppe: [],
      grace_period_seconds: parseFloat(zoneGraceInput.value) || 3.0,
      alert_severity: zoneSeveritySelect.value,
      description: `Hospital designated area: ${zoneNameInput.value}`
    };

    try {
      const res = await fetch("/api/zones", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        activeEditingZoneId = null;
        currentPoints = [];
        deleteZoneBtn.style.display = "none";
        document.getElementById("zone-form").reset();
        await fetchZones();
      } else {
        alert("Failed to save zone policy.");
      }
    } catch (e) {
      console.error("Error saving zone:", e);
    }
  });

  // Delete Zone
  deleteZoneBtn.addEventListener("click", async () => {
    if (!activeEditingZoneId) return;
    if (!confirm(`Delete zone '${zoneNameInput.value}'?`)) return;

    try {
      const res = await fetch(`/api/zones/${activeEditingZoneId}`, { method: "DELETE" });
      if (res.ok) {
        activeEditingZoneId = null;
        currentPoints = [];
        deleteZoneBtn.style.display = "none";
        document.getElementById("zone-form").reset();
        await fetchZones();
      }
    } catch (e) {
      console.error("Error deleting zone:", e);
    }
  });

  // Initial load
  loadBackgroundFrame();
  fetchZones();
});

