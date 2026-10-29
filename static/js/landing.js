// Landing Tanya Padi: navigasi kaca saat digulir, bagian muncul saat terlihat,
// dan angka penelitian dihitung naik sekali.
(function () {
  const nav = document.getElementById('navUtama');
  const aturNav = () => nav.classList.toggle('tergulir', window.scrollY > 12);
  aturNav();
  window.addEventListener('scroll', aturNav, { passive: true });

  // Menu HP tertutup setelah salah satu tautan diketuk
  const menu = document.getElementById('menuUtama');
  menu.querySelectorAll('.nav-link').forEach((a) => a.addEventListener('click', () => {
    if (menu.classList.contains('show') && window.bootstrap) bootstrap.Collapse.getOrCreateInstance(menu).hide();
  }));

  const geraknyaDikurangi = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const formatAngka = (n) => n.toLocaleString('id-ID');

  function hitungNaik(el) {
    const target = Number(el.dataset.hitung);
    if (geraknyaDikurangi || !target) { el.textContent = formatAngka(target); return; }
    const mulai = performance.now(), lama = 1300;
    const langkah = (t) => {
      const p = Math.min(1, (t - mulai) / lama);
      const mulus = 1 - Math.pow(1 - p, 3);
      el.textContent = formatAngka(Math.round(target * mulus));
      if (p < 1) requestAnimationFrame(langkah);
    };
    requestAnimationFrame(langkah);
  }

  const bagian = document.querySelectorAll('[data-muncul]');
  const angka = document.querySelectorAll('[data-hitung]');
  if (!('IntersectionObserver' in window)) {
    bagian.forEach((el) => el.classList.add('terlihat'));
    return;
  }
  const pengamat = new IntersectionObserver((entri) => {
    entri.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add('terlihat');
      e.target.querySelectorAll('[data-hitung]').forEach(hitungNaik);
      pengamat.unobserve(e.target);
    });
  }, { threshold: 0.18, rootMargin: '0px 0px -40px 0px' });
  bagian.forEach((el) => pengamat.observe(el));
  // Angka di luar [data-muncul] tetap dihitung
  angka.forEach((el) => { if (!el.closest('[data-muncul]')) hitungNaik(el); });
})();
