// Hospital PPE Compliance Analytics Visualizations with Chart.js
document.addEventListener("DOMContentLoaded", () => {
  if (typeof analyticsData === "undefined") return;

  const darkThemeDefaults = {
    color: "#94a3b8",
    font: {
      family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    }
  };
  Chart.defaults.color = darkThemeDefaults.color;

  // 1. Missing PPE Distribution Chart (Doughnut)
  const itemLabels = Object.keys(analyticsData.missing_item_distribution || {});
  const itemValues = Object.values(analyticsData.missing_item_distribution || {});

  const itemsCtx = document.getElementById("items-chart");
  if (itemsCtx) {
    new Chart(itemsCtx, {
      type: "doughnut",
      data: {
        labels: itemLabels.length > 0 ? itemLabels : ["No Violations"],
        datasets: [{
          data: itemValues.length > 0 ? itemValues : [1],
          backgroundColor: [
            "#ef4444", "#f97316", "#f59e0b", "#06b6d4", "#3b82f6", "#a855f7"
          ],
          borderWidth: 1,
          borderColor: "#182234"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "right" }
        }
      }
    });
  }

  // 2. Violations by Zone Chart (Bar)
  const zoneLabels = Object.keys(analyticsData.zone_distribution || {});
  const zoneValues = Object.values(analyticsData.zone_distribution || {});

  const zonesCtx = document.getElementById("zones-chart");
  if (zonesCtx) {
    new Chart(zonesCtx, {
      type: "bar",
      data: {
        labels: zoneLabels.length > 0 ? zoneLabels : ["No Data"],
        datasets: [{
          label: "Violations Recorded",
          data: zoneValues.length > 0 ? zoneValues : [0],
          backgroundColor: "rgba(239, 68, 68, 0.7)",
          borderColor: "#ef4444",
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, grid: { color: "#334155" } },
          x: { grid: { display: false } }
        }
      }
    });
  }

  // 3. Inspection Status Breakdown (Pie)
  const breakdownCtx = document.getElementById("breakdown-chart");
  if (breakdownCtx) {
    new Chart(breakdownCtx, {
      type: "pie",
      data: {
        labels: ["Compliant", "Partially Compliant", "Violations"],
        datasets: [{
          data: [
            analyticsData.compliant_evaluations || 0,
            analyticsData.partial_evaluations || 0,
            analyticsData.violation_evaluations || 0
          ],
          backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"],
          borderWidth: 1,
          borderColor: "#182234"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "right" }
        }
      }
    });
  }

  // 4. Incident Severity Breakdown (Bar)
  const severityLabels = Object.keys(analyticsData.severity_distribution || {});
  const severityValues = Object.values(analyticsData.severity_distribution || {});

  const severityCtx = document.getElementById("severity-chart");
  if (severityCtx) {
    new Chart(severityCtx, {
      type: "bar",
      data: {
        labels: severityLabels,
        datasets: [{
          label: "Incident Count",
          data: severityValues,
          backgroundColor: [
            "rgba(239, 68, 68, 0.8)",  // Critical
            "rgba(245, 158, 11, 0.8)",  // Warning
            "rgba(6, 182, 212, 0.8)"    // Info
          ],
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, grid: { color: "#334155" } },
          x: { grid: { display: false } }
        }
      }
    });
  }
});

