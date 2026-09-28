// Incident Log Table Filtering, Evidence Modal, Status Transitions & CSV Export
document.addEventListener("DOMContentLoaded", () => {
  const searchInput = document.getElementById("incident-search");
  const severityFilter = document.getElementById("severity-filter");
  const statusFilter = document.getElementById("status-filter");
  const exportCsvBtn = document.getElementById("export-csv-btn");
  const cleanupBtn = document.getElementById("cleanup-btn");
  const tableBody = document.getElementById("incidents-table-body");
  const snapshotModal = document.getElementById("snapshot-modal");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-title");
  const modalDetails = document.getElementById("modal-details");

  // Open modal
  window.openSnapshotModal = (incidentId, filename, zoneName, details, status) => {
    modalTitle.textContent = `Evidence Snapshot: ${incidentId} [${status || 'NEW'}] (${zoneName})`;
    modalImg.src = `/api/incidents/${incidentId}/snapshot`;
    modalDetails.innerHTML = `<strong>Details:</strong> ${details || 'None'}<br><small style="color: #93c5fd; margin-top: 4px; display: inline-block;">(Facial region blurred for HIPAA / Clinical Privacy compliance)</small>`;
    snapshotModal.classList.add("open");
  };

  // Close modal
  window.closeSnapshotModal = () => {
    snapshotModal.classList.remove("open");
  };

  snapshotModal.addEventListener("click", (e) => {
    if (e.target === snapshotModal) {
      closeSnapshotModal();
    }
  });

  // Transition Incident Status
  window.updateStatus = async (incidentId, newStatus) => {
    try {
      const res = await fetch(`/api/incidents/${incidentId}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          status: newStatus,
          user: "Nurse Station / Clinical Safety",
          notes: `Status transitioned to ${newStatus}`
        })
      });
      const data = await res.json();
      if (data.status === "success") {
        // Refresh page or update row badge
        window.location.reload();
      } else {
        alert("Failed to update status: " + (data.detail || "Unknown error"));
      }
    } catch (err) {
      alert("Error updating incident status: " + err);
    }
  };

  // Retention Cleanup
  if (cleanupBtn) {
    cleanupBtn.addEventListener("click", async () => {
      if (!confirm("Enforce 7-day snapshot and 30-day incident retention cleanup now?")) return;
      try {
        const res = await fetch("/api/incidents/cleanup", { method: "POST" });
        const data = await res.json();
        alert(`Retention enforced: Purged ${data.purged_snapshots} old snapshots and ${data.purged_incidents} expired incidents.`);
        window.location.reload();
      } catch (err) {
        alert("Retention cleanup error: " + err);
      }
    });
  }

  // Filter Table
  function filterTable() {
    const query = searchInput.value.toLowerCase();
    const severity = severityFilter.value;
    const status = statusFilter ? statusFilter.value : "ALL";
    const rows = tableBody.querySelectorAll("tr");

    rows.forEach(row => {
      const text = row.textContent.toLowerCase();
      const matchesQuery = !query || text.includes(query);
      const matchesSeverity = severity === "ALL" || text.includes(severity.toLowerCase());
      const matchesStatus = status === "ALL" || text.includes(status.toLowerCase());

      row.style.display = (matchesQuery && matchesSeverity && matchesStatus) ? "" : "none";
    });
  }

  searchInput.addEventListener("input", filterTable);
  severityFilter.addEventListener("change", filterTable);
  if (statusFilter) {
    statusFilter.addEventListener("change", filterTable);
  }

  // Export CSV
  exportCsvBtn.addEventListener("click", () => {
    const rows = Array.from(tableBody.querySelectorAll("tr"));
    if (rows.length === 0) return;

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Episode ID,Status,Timestamp,Zone,Severity,Track ID,Violations,Duration\n";

    rows.forEach(r => {
      const cols = Array.from(r.querySelectorAll("td")).map(td => `"${td.innerText.replace(/"/g, '""').trim()}"`);
      if (cols.length >= 8) {
        csvContent += cols.slice(0, 8).join(",") + "\n";
      }
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `ppe_episodes_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
});
