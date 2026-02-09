/* Geometric Deep Learning Study Guide - Shared Scripts */

// Theme toggle
function initTheme() {
  const saved = localStorage.getItem('gdl-theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeButton(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('gdl-theme', next);
  updateThemeButton(next);
}

function updateThemeButton(theme) {
  const btn = document.querySelector('.theme-toggle');
  if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
}

// Sidebar toggle (mobile)
function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  if (sidebar) sidebar.classList.toggle('open');
}

// Reading progress
function initProgress() {
  const bar = document.querySelector('.progress-bar');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrolled = window.scrollY;
    bar.style.width = Math.min(100, (scrolled / docHeight) * 100) + '%';
  });
}

// Active sidebar section highlighting
function initSidebarHighlight() {
  const links = document.querySelectorAll('.sidebar a[href^="#"]');
  if (links.length === 0) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const active = document.querySelector(`.sidebar a[href="#${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-80px 0px -70% 0px' });

  links.forEach(link => {
    const id = link.getAttribute('href').slice(1);
    const target = document.getElementById(id);
    if (target) observer.observe(target);
  });
}

// Close sidebar on link click (mobile)
function initSidebarLinkClose() {
  const links = document.querySelectorAll('.sidebar a');
  links.forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 1024) {
        document.querySelector('.sidebar')?.classList.remove('open');
      }
    });
  });
}

// Init all
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initProgress();
  initSidebarHighlight();
  initSidebarLinkClose();
});
