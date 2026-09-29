// SafeCare Common Navigation & UI Helper Script
document.addEventListener("DOMContentLoaded", () => {
  // 1. Live Clock Initialization
  const clockEl = document.getElementById("live-clock");
  if (clockEl) {
    function updateClock() {
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

      clockEl.textContent = `${day}, ${date} ${month} ${year} ${hours}:${minutes}:${seconds} ${ampm}`;
    }
    updateClock();
    setInterval(updateClock, 1000);
  }

  // 2. Incident Badge Live Polling
  async function updateIncidentBadges() {
    try {
      const res = await fetch("/api/dashboard/summary");
      if (res.ok) {
        const data = await res.json();
        const count = data.total_incidents_today || 0;
        const topbarCount = document.getElementById("topbar-notif-count");
        const sidebarCount = document.getElementById("sidebar-incident-badge");
        if (topbarCount) topbarCount.textContent = count;
        if (sidebarCount) sidebarCount.textContent = count;
      }
    } catch (e) {
      // Fallback silently if offline
    }
  }
  updateIncidentBadges();
  setInterval(updateIncidentBadges, 10000);

  // 3. Mobile Sidebar Toggle
  const toggleBtn = document.getElementById("sidebar-toggle-btn");
  const sidebar = document.querySelector(".safecare-sidebar");
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("mobile-open");
    });

    // Close when clicking outside on mobile
    document.addEventListener("click", (e) => {
      if (sidebar.classList.contains("mobile-open") && !sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
        sidebar.classList.remove("mobile-open");
      }
    });
  }

  // 4. Modal Safe Defaults
  if (typeof window.openCameraModal === "undefined") {
    window.openCameraModal = function() {
      if (typeof window.openAddCameraModal === "function") {
        window.openAddCameraModal();
      }
    };
  }

  // 5. Global Modal Backdrop & Esc Key Dismissal
  document.addEventListener("click", (e) => {
    if (e.target.classList && e.target.classList.contains("modal-backdrop") && e.target.classList.contains("open")) {
      e.target.classList.remove("open");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      document.querySelectorAll(".modal-backdrop.open").forEach(m => m.classList.remove("open"));
    }
  });

  // 6. Direct click listener for Insert/Change Video button
  const changeVideoBtn = document.querySelector(".btn-change-video");
  if (changeVideoBtn) {
    changeVideoBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const modal = document.getElementById("insert-video-modal");
      if (modal) {
        modal.classList.add("open");
        modal.style.display = "flex";
        modal.style.opacity = "1";
        modal.style.pointerEvents = "auto";
        modal.style.visibility = "visible";
        modal.style.zIndex = "99999";
      }
      if (typeof window.switchVideoModalTab === "function") window.switchVideoModalTab("demos");
      if (typeof loadSampleVideosList === "function") loadSampleVideosList();
    });
  }
});

