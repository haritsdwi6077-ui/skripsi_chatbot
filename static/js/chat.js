// Halaman tanya jawab Tanya Padi. Riwayat percakapan disimpan di localStorage
// browser; server tetap stateless. Komponen laci, dialog, dan tombol memakai Bootstrap 5.3.
const TAMPILKAN_MODE = document.body.dataset.tampilkanMode === '1';
const CONTOH = [
  { judul: 'Penyakit karena jamur', tanya: [
    'Apa gejala penyakit blas pada daun padi?',
    'Seperti apa gejala bercak cokelat pada padi?',
    'Bagaimana cara mencegah blas leher malai?' ] },
  { judul: 'Penyakit karena bakteri dan virus', tanya: [
    'Daun padi mengering dari ujung dan pinggirnya, apakah itu kresek?',
    'Daun padi menguning oranye dan tanaman kerdil, apakah itu tungro?' ] },
  { judul: 'Hama', tanya: [
    'Apa tanda serangan wereng batang cokelat?',
    'Apa itu sundep dan beluk?',
    'Bagaimana cara mengendalikan keong mas di sawah?' ] },
  { judul: 'Gambaran umum', tanya: [
    'Penyakit padi apa saja yang disebabkan jamur?',
    'Sebutkan hama utama tanaman padi' ] },
];
const MAKS_RIWAYAT = 3;          // giliran terakhir yang dikirim ke server untuk pertanyaan lanjutan
const MAKS_PERCAKAPAN = 40;      // percakapan lama yang disimpan di perangkat
const KUNCI = 'tanyaPadi.percakapan.v1';
const KUNCI_DRAF = 'tanyaPadi.drafBaru';
const NAMA_MODE = { rag: 'Dengan RAG', llm: 'Tanpa RAG' };
const PETUNJUK = {
  rag: 'Jawaban bisa kurang lengkap. Periksa sumbernya.',
  llm: 'Mode pembanding: jawaban tanpa dokumen rujukan dan tanpa sumber.'
};

const $ = (id) => document.getElementById(id);
const isi = $('isi'), sambutan = $('sambutan'), elPercakapan = $('percakapan');
const form = $('form'), input = $('pertanyaan'), tombolKirim = $('kirim');
const samping = $('samping'), daftarRiwayat = $('daftarRiwayat'), cariRiwayat = $('cariRiwayat');
const panduan = $('panduan');
const bs = window.bootstrap;
const radioMode = document.querySelectorAll('input[name="mode"]');

// ---------- Penyimpanan di perangkat (aman bila localStorage tidak tersedia) ----------
const simpan = {
  baca(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  tulis(k, v) { try { localStorage.setItem(k, v); return true; } catch (e) { return false; } },
  hapus(k) { try { localStorage.removeItem(k); } catch (e) {} },
};
function muatData() {
  try {
    const d = JSON.parse(simpan.baca(KUNCI) || 'null');
    if (d && Array.isArray(d.daftar)) return d;
  } catch (e) {}
  return { versi: 1, aktif: null, daftar: [] };
}
let data = muatData();
function simpanData() {
  data.daftar.sort((a, b) => b.diubah - a.diubah);
  data.daftar = data.daftar.slice(0, MAKS_PERCAKAPAN);
  // Bila penyimpanan penuh, percakapan paling lama dibuang sampai muat
  while (!simpan.tulis(KUNCI, JSON.stringify(data)) && data.daftar.length > 1) data.daftar.pop();
}
const buatId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
function percakapanBaru(mode) {
  return { id: buatId(), judul: '', dibuat: Date.now(), diubah: Date.now(), mode: mode || 'rag', pesan: [], riwayat: [], draf: '' };
}
const tersimpan = (pc) => data.daftar.includes(pc);

const modeAwal = TAMPILKAN_MODE && new URLSearchParams(location.search).get('mode') === 'llm' ? 'llm' : 'rag';
let aktif = data.daftar.find((p) => p.id === data.aktif) || percakapanBaru(modeAwal);
// Alamat yang menyebut mode (misalnya /tanya?mode=llm) membuka percakapan baru dalam mode itu
if (TAMPILKAN_MODE && new URLSearchParams(location.search).has('mode') && aktif.mode !== modeAwal) {
  if (aktif.pesan.length) aktif = percakapanBaru(modeAwal); else aktif.mode = modeAwal;
}
if (!tersimpan(aktif)) aktif.draf = simpan.baca(KUNCI_DRAF) || '';
let sibuk = false;

// ---------- Pengubah teks jawaban (format markdown sederhana) menjadi HTML yang aman ----------
function escapeHtml(teks) {
  return String(teks).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
const rapikanSitasi = (t) => String(t || '').replace(/【\s*(\d+)[^】]*】/g, '[$1]');   // gaya sitasi gpt-oss: 【1】 menjadi [1]
function rapikan(teks) {
  const inline = (s) => escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m, n) =>
      n.split(',').map((x) => '<span class="rujuk" title="Sumber nomor ' + x.trim() + '">' + x.trim() + '</span>').join(''));
  const baris = rapikanSitasi(String(teks).replace(/\r/g, '').replace(/[‑‐]/g, '-').replace(/ /g, ' ')).split('\n');
  const pecahBr = (s) => s.split(/<br\s*\/?>/i).map((x) => x.trim()).filter(Boolean);
  let html = '', daftar = null, paragraf = [], tabel = [];
  const tutupTabel = () => {
    if (!tabel.length) return;
    const sel = (r) => r.replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
    const pemisah = (r) => /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(r);
    const isiSel = (c) => pecahBr(c).map(inline).join('<br>');
    const punyaJudul = tabel.length > 1 && pemisah(tabel[1]);
    const baris2 = tabel.filter((r) => !pemisah(r));
    let t = '<div class="tabel-bungkus"><table>';
    if (punyaJudul) t += '<thead><tr>' + sel(baris2.shift()).map((c) => '<th>' + isiSel(c) + '</th>').join('') + '</tr></thead>';
    t += '<tbody>' + baris2.map((r) => '<tr>' + sel(r).map((c) => '<td>' + isiSel(c) + '</td>').join('') + '</tr>').join('') + '</tbody></table></div>';
    html += t; tabel = [];
  };
  const tutupParagraf = () => { if (paragraf.length) { html += '<p>' + paragraf.map(inline).join('<br>') + '</p>'; paragraf = []; } };
  const tutupDaftar = () => { if (daftar) { html += '</' + daftar + '>'; daftar = null; } };
  for (const b of baris) {
    const s = b.trim();
    const anak = /^(\s{2,}|\t)/.test(b) ? ' class="anak"' : '';
    let m;
    if (s.startsWith('|')) { tutupParagraf(); tutupDaftar(); tabel.push(s); continue; }
    tutupTabel();
    if (!s || /^(-{3,}|\*{3,}|_{3,})$/.test(s)) { tutupParagraf(); tutupDaftar(); continue; }
    if ((m = s.match(/^#{1,6}\s+(.*)$/))) { tutupParagraf(); tutupDaftar(); html += '<h4>' + inline(m[1]) + '</h4>'; continue; }
    if ((m = s.match(/^[-*•]\s+(.*)$/))) { tutupParagraf(); if (daftar !== 'ul') { tutupDaftar(); html += '<ul>'; daftar = 'ul'; } html += '<li' + anak + '>' + pecahBr(m[1]).map(inline).join('<br>') + '</li>'; continue; }
    if ((m = s.match(/^\d+[.)]\s+(.*)$/))) { tutupParagraf(); if (daftar !== 'ol') { tutupDaftar(); html += '<ol>'; daftar = 'ol'; } html += '<li' + anak + '>' + pecahBr(m[1]).map(inline).join('<br>') + '</li>'; continue; }
    tutupDaftar(); paragraf.push(...pecahBr(s));
  }
  tutupTabel(); tutupParagraf(); tutupDaftar();
  return html || '<p></p>';
}

// ---------- Contoh pertanyaan ----------
const wadahContoh = $('contoh');
CONTOH.forEach((k) => {
  const g = document.createElement('div');
  const h = document.createElement('h3'); h.textContent = k.judul; g.appendChild(h);
  const chips = document.createElement('div'); chips.className = 'chips';
  k.tanya.forEach((t) => chips.appendChild(buatChip(t)));
  g.appendChild(chips); wadahContoh.appendChild(g);
});
function buatChip(teks) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'chip'; b.textContent = teks;
  b.addEventListener('click', () => kirim(teks));
  return b;
}

// ---------- Menampilkan pesan ----------
function buatPesan(p, terakhir) {
  const el = document.createElement('div');
  if (p.jenis === 'pengguna') { el.className = 'pesan pengguna'; el.textContent = p.teks; return el; }
  if (p.jenis === 'pemisah') { el.className = 'pesan pemisah'; el.textContent = p.teks; return el; }
  if (p.jenis === 'galat') {
    el.className = 'pesan galat'; el.setAttribute('role', 'alert');
    const t = document.createElement('span'); t.textContent = p.teks; el.appendChild(t);
    if (p.tanya && terakhir) {
      const ulang = document.createElement('button');
      ulang.type = 'button'; ulang.className = 'aksi-kecil';
      ulang.innerHTML = '<svg class="ikon" aria-hidden="true"><use href="#i-ulang"/></svg>Kirim ulang pertanyaan';
      ulang.addEventListener('click', () => kirimUlang(p));
      el.appendChild(ulang);
    }
    return el;
  }
  // jawaban sistem
  const d = p.data || {};
  const modeJawab = d.mode === 'llm' ? 'llm' : 'rag';
  el.className = 'pesan sistem';
  let html = '<div class="kepala-jawaban"><svg class="bulir" aria-hidden="true"><use href="#bulir"/></svg><b>Tanya Padi</b>';
  if (TAMPILKAN_MODE) html += '<span class="lencana-mode ' + modeJawab + '">' + NAMA_MODE[modeJawab] + '</span>';
  html += '</div>';
  const tafsir = (d.standalone_question || '').trim();
  if (tafsir && tafsir !== (p.tanya || '').trim()) html += '<p class="tafsir">Ditafsirkan sebagai: <span>' + escapeHtml(tafsir) + '</span></p>';
  html += '<div class="isi-jawaban">' + rapikan(d.answer || '') + '</div>';
  const sumber = d.sources || [];
  const detik = d.timing && typeof d.timing.total_s === 'number' ? d.timing.total_s : null;
  html += '<div class="kaki">';
  if (sumber.length) {
    // Hanya sumber yang dikutip di jawaban; nomornya sama dengan lencana di jawaban
    html += '<details><summary><svg class="ikon" aria-hidden="true"><use href="#i-buku"/></svg>Sumber rujukan (' + sumber.length + ')</summary><ul class="daftar-sumber">' +
      sumber.map((s) => '<li>' + escapeHtml(s).replace(/^\[(\d+)\]\s*(.*)$/, '<span class="rujuk">$1</span><span>$2</span>') + '</li>').join('') + '</ul></details>';
  }
  html += '<div class="kaki-baris">' +
    '<button type="button" class="aksi-kecil salin"><svg class="ikon" aria-hidden="true"><use href="#i-salin"/></svg><span>Salin jawaban</span></button>' +
    (detik !== null ? '<span>Dijawab dalam ' + detik.toFixed(1).replace('.', ',') + ' detik</span>' : '') + '</div>';
  html += '</div>';
  el.innerHTML = html;
  el.querySelector('.salin').addEventListener('click', (e) => {
    const label = e.currentTarget.querySelector('span');
    const selesai = () => { label.textContent = 'Jawaban disalin'; setTimeout(() => (label.textContent = 'Salin jawaban'), 2000); };
    if (navigator.clipboard) navigator.clipboard.writeText(rapikanSitasi(d.answer)).then(selesai).catch(() => {});
  });
  const saran = (d.suggestions || []).filter(Boolean);
  if (terakhir && saran.length) {
    const s = document.createElement('div');
    s.className = 'saran';
    s.innerHTML = '<p>Pertanyaan lanjutan</p>';
    const chips = document.createElement('div'); chips.className = 'chips';
    saran.forEach((t) => chips.appendChild(buatChip(t)));
    s.appendChild(chips);
    el.querySelector('.kaki').appendChild(s);
  }
  return el;
}

function tampilkanPercakapan() {
  elPercakapan.innerHTML = '';
  const n = aktif.pesan.length;
  aktif.pesan.forEach((p, i) => elPercakapan.appendChild(buatPesan(p, i === n - 1)));
  sambutan.hidden = n > 0;
  aturModeUI(aktif.mode);
  input.value = aktif.draf || '';
  sesuaikanTinggi();
  requestAnimationFrame(() => { isi.scrollTop = n ? isi.scrollHeight : 0; });
  tampilkanRiwayat();
  tampilkanLanjutan();
}
function waktuSingkat(t) {
  const g = kelompokWaktu(t);
  const jam = new Date(t).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
  if (g === 'Hari ini' || g === 'Kemarin') return g + ', ' + jam;
  return new Date(t).toLocaleDateString('id-ID', { day: 'numeric', month: 'short' });
}
function tampilkanLanjutan() {
  const lama = data.daftar.filter((pc) => pc !== aktif && pc.pesan.length).slice(0, 3);
  const wadah = $('daftarLanjut');
  wadah.innerHTML = '';
  $('lanjutkan').hidden = !lama.length;
  lama.forEach((pc) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'lanjut';
    b.innerHTML = '<strong></strong><small></small>';
    b.querySelector('strong').textContent = pc.judul || 'Percakapan tanpa judul';
    const n = pc.pesan.filter((p) => p.jenis === 'pengguna').length;
    b.querySelector('small').textContent = waktuSingkat(pc.diubah) + ', ' + n + ' pertanyaan';
    b.addEventListener('click', () => bukaPercakapan(pc));
    wadah.appendChild(b);
  });
}
function tambahEl(el) {
  elPercakapan.appendChild(el);
  sambutan.hidden = true;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  return el;
}
function hapusSaranLama() {
  elPercakapan.querySelectorAll('.saran, .pesan.galat .aksi-kecil').forEach((x) => x.remove());
}

// ---------- Mengirim pertanyaan ----------
async function kirim(teks) {
  teks = (teks || '').trim();
  if (!teks || sibuk) return;
  const pc = aktif;
  if (!tersimpan(pc)) { data.daftar.unshift(pc); simpan.hapus(KUNCI_DRAF); }
  if (!pc.judul) pc.judul = teks.length > 80 ? teks.slice(0, 78).trimEnd() + '…' : teks;
  pc.pesan.push({ jenis: 'pengguna', teks });
  pc.draf = ''; pc.diubah = Date.now(); data.aktif = pc.id;
  simpanData();
  hapusSaranLama();
  tambahEl(buatPesan(pc.pesan[pc.pesan.length - 1], false));
  input.value = ''; sesuaikanTinggi();
  tutupRiwayat();
  tampilkanRiwayat();

  sibuk = true; tombolKirim.disabled = true;
  const tunggu = document.createElement('div');
  tunggu.className = 'pesan sistem menunggu';
  tunggu.innerHTML = '<span class="titik" aria-hidden="true"><i></i><i></i><i></i></span><span>Sedang diproses…</span>';
  tambahEl(tunggu);
  // Jawaban bisa butuh 5 sampai 20 detik; beri tanda bahwa aplikasi tidak macet
  const jedaTunggu = setTimeout(() => { tunggu.lastChild.textContent = 'Masih diproses, mohon tunggu sebentar…'; }, 8000);

  let pesanBaru;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: teks, history: pc.riwayat.slice(-MAKS_RIWAYAT), mode: pc.mode })
    });
    let d = {};
    try { d = await res.json(); } catch (e) {}
    if (!res.ok) {
      pesanBaru = { jenis: 'galat', tanya: teks, teks: d.error || 'Server tidak dapat memproses pertanyaan ini. Coba kirim ulang beberapa saat lagi.' };
    } else {
      pc.riwayat.push({ question: teks, answer: d.answer || '' });
      pc.riwayat = pc.riwayat.slice(-MAKS_RIWAYAT);
      pesanBaru = { jenis: 'sistem', tanya: teks, data: {
        answer: d.answer || '', mode: d.mode, sources: d.sources || [], standalone_question: d.standalone_question || '',
        timing: d.timing || {}, suggestions: d.suggestions || [] } };
    }
  } catch (err) {
    pesanBaru = { jenis: 'galat', tanya: teks, teks: 'Tidak tersambung ke server. Periksa sinyal atau koneksi internet, lalu kirim ulang pertanyaan.' };
  } finally {
    clearTimeout(jedaTunggu);
    tunggu.remove();
    sibuk = false; tombolKirim.disabled = false;
  }
  pc.pesan.push(pesanBaru);
  pc.diubah = Date.now();
  simpanData();
  if (pc === aktif) {
    const el = tambahEl(buatPesan(pesanBaru, true));
    if (pesanBaru.jenis === 'sistem') el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (matchMedia('(hover: hover)').matches) input.focus();
  }
  tampilkanRiwayat();
}
function kirimUlang(p) {
  const i = aktif.pesan.indexOf(p);
  if (i < 0 || sibuk) return;
  const jumlah = i > 0 && aktif.pesan[i - 1].jenis === 'pengguna' && aktif.pesan[i - 1].teks === p.tanya ? 2 : 1;
  aktif.pesan.splice(i - jumlah + 1, jumlah);
  simpanData();
  tampilkanPercakapan();
  kirim(p.tanya);
}
form.addEventListener('submit', (e) => { e.preventDefault(); kirim(input.value); });

// ---------- Kolom tulis: tinggi ikut isi, Enter = kirim, draf disimpan ----------
function sesuaikanTinggi() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 144) + 'px'; }
let jedaDraf;
input.addEventListener('input', () => {
  sesuaikanTinggi();
  aktif.draf = input.value;
  clearTimeout(jedaDraf);
  jedaDraf = setTimeout(() => { if (tersimpan(aktif)) simpanData(); else simpan.tulis(KUNCI_DRAF, input.value); }, 400);
});
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
});

// ---------- Mode jawaban (pembanding penelitian) ----------
const petunjuk = $('petunjuk');
function aturModeUI(m) {
  radioMode.forEach((r) => { r.checked = r.value === m; });
  petunjuk.textContent = PETUNJUK[m] || PETUNJUK.rag;
}
radioMode.forEach((r) => r.addEventListener('change', () => {
  if (r.value === aktif.mode) return;
  aktif.mode = r.value;
  // Riwayat tidak dibawa lintas mode, agar jawaban tiap mode tidak saling memengaruhi
  aktif.riwayat = [];
  if (aktif.pesan.length) {
    const p = { jenis: 'pemisah', teks: 'Mode diganti ke ' + NAMA_MODE[aktif.mode] + ', percakapan dimulai ulang' };
    aktif.pesan.push(p);
    hapusSaranLama();
    tambahEl(buatPesan(p, true));
    simpanData();
  }
  aturModeUI(aktif.mode);
}));

// ---------- Riwayat percakapan ----------
function kelompokWaktu(t) {
  const hariIni = new Date(); hariIni.setHours(0, 0, 0, 0);
  const selisih = (hariIni - new Date(t).setHours(0, 0, 0, 0)) / 86400000;
  if (selisih <= 0) return 'Hari ini';
  if (selisih === 1) return 'Kemarin';
  if (selisih < 7) return '7 hari terakhir';
  return 'Lebih lama';
}
function tampilkanRiwayat() {
  const kata = cariRiwayat.value.trim().toLowerCase();
  const cocok = data.daftar.filter((pc) => !kata ||
    pc.pesan.some((p) => p.jenis === 'pengguna' && p.teks.toLowerCase().includes(kata)));
  daftarRiwayat.innerHTML = '';
  if (!cocok.length) {
    const p = document.createElement('p'); p.className = 'kosong-riwayat';
    p.textContent = kata ? 'Tidak ada pertanyaan lama yang cocok.' : 'Belum ada percakapan. Pertanyaan yang Anda kirim akan tersimpan di sini.';
    daftarRiwayat.appendChild(p); return;
  }
  let grup = null, judulGrup = '';
  cocok.forEach((pc) => {
    const g = kelompokWaktu(pc.diubah);
    if (g !== judulGrup) {
      judulGrup = g; grup = document.createElement('div'); grup.className = 'grup-riwayat';
      const h = document.createElement('h3'); h.textContent = g; grup.appendChild(h);
      daftarRiwayat.appendChild(grup);
    }
    const jumlah = pc.pesan.filter((p) => p.jenis === 'pengguna').length;
    const item = document.createElement('div');
    item.className = 'item-riwayat';
    if (pc === aktif) item.setAttribute('aria-current', 'true');
    const buka = document.createElement('button');
    buka.type = 'button'; buka.className = 'buka-item';
    buka.innerHTML = '<strong></strong><small></small>';
    buka.querySelector('strong').textContent = pc.judul || 'Percakapan tanpa judul';
    buka.querySelector('small').textContent = jumlah + ' pertanyaan' + (TAMPILKAN_MODE && pc.mode === 'llm' ? ', tanpa RAG' : '');
    buka.title = pc.judul;
    buka.addEventListener('click', () => bukaPercakapan(pc));
    const hapus = document.createElement('button');
    hapus.type = 'button'; hapus.className = 'hapus-item';
    hapus.setAttribute('aria-label', 'Hapus percakapan: ' + (pc.judul || 'tanpa judul'));
    hapus.innerHTML = '<svg class="ikon" aria-hidden="true"><use href="#i-hapus"/></svg>';
    hapus.addEventListener('click', () => hapusPercakapan(pc));
    item.append(buka, hapus);
    grup.appendChild(item);
  });
}
function bukaPercakapan(pc) {
  if (sibuk) return;
  aktif = pc; data.aktif = pc.id; simpanData();
  tampilkanPercakapan();
  tutupRiwayat();
}
function mulaiBaru() {
  if (sibuk) return;
  aktif = percakapanBaru(aktif.mode);
  data.aktif = null; simpanData();
  tampilkanPercakapan();
  tutupRiwayat();
  if (matchMedia('(hover: hover)').matches) input.focus();   // di HP keyboard tidak langsung muncul
}
function hapusPercakapan(pc) {
  if (!confirm('Hapus percakapan "' + (pc.judul || 'tanpa judul') + '" dari perangkat ini?')) return;
  data.daftar = data.daftar.filter((x) => x !== pc);
  if (pc === aktif) { aktif = percakapanBaru(pc.mode); data.aktif = null; simpanData(); tampilkanPercakapan(); }
  else { simpanData(); tampilkanRiwayat(); }
}
$('hapusSemua').addEventListener('click', () => {
  if (!data.daftar.length || !confirm('Hapus semua riwayat percakapan di perangkat ini?')) return;
  data = { versi: 1, aktif: null, daftar: [] };
  aktif = percakapanBaru(aktif.mode);
  simpanData(); tampilkanPercakapan();
});
$('tombolBaru').addEventListener('click', mulaiBaru);
cariRiwayat.addEventListener('input', tampilkanRiwayat);

// Riwayat di HP: laci Bootstrap (offcanvas-lg); di laptop tampil tetap di samping
const laci = () => (bs ? bs.Offcanvas.getOrCreateInstance(samping) : null);
function bukaRiwayat() { if (laci()) laci().show(); }
function tutupRiwayat() { if (samping.classList.contains('show') && laci()) laci().hide(); }
$('lihatSemuaRiwayat').addEventListener('click', bukaRiwayat);

// ---------- Panduan, tema, ukuran huruf ----------
const bukaPanduan = () => { if (bs) bs.Modal.getOrCreateInstance(panduan).show(); };
$('tombolPanduan').addEventListener('click', bukaPanduan);
$('bacaPanduan').addEventListener('click', bukaPanduan);
$('mulaiBertanya').addEventListener('click', () => input.focus());

const tombolTema = $('tombolTema');
function aturTema(t) {
  document.documentElement.dataset.tema = t;
  document.documentElement.setAttribute('data-bs-theme', t === 'gelap' ? 'dark' : 'light');
  tombolTema.querySelector('use').setAttribute('href', t === 'gelap' ? '#i-matahari' : '#i-bulan');
  tombolTema.setAttribute('aria-label', t === 'gelap' ? 'Ganti ke tema terang' : 'Ganti ke tema gelap');
  tombolTema.title = tombolTema.getAttribute('aria-label');
  document.querySelector('meta[name="theme-color"]').content = t === 'gelap' ? '#121610' : '#F4EEDF';
}
aturTema(document.documentElement.dataset.tema === 'terang' ? 'terang' : 'gelap');
tombolTema.addEventListener('click', () => {
  const t = document.documentElement.dataset.tema === 'gelap' ? 'terang' : 'gelap';
  aturTema(t); simpan.tulis('tanyaPadi.tema', t);
});

const tombolHuruf = $('tombolHuruf');
function aturHuruf(besar) {
  document.documentElement.style.setProperty('--skala', besar ? '1.2' : '1');
  tombolHuruf.setAttribute('aria-pressed', String(besar));
  tombolHuruf.querySelector('span').textContent = besar ? 'A−' : 'A+';
  tombolHuruf.setAttribute('aria-label', besar ? 'Kembalikan ukuran huruf' : 'Perbesar huruf');
  tombolHuruf.title = tombolHuruf.getAttribute('aria-label');
  simpan.tulis('tanyaPadiHurufBesar', besar ? '1' : '0');
}
aturHuruf(simpan.baca('tanyaPadiHurufBesar') === '1');
tombolHuruf.addEventListener('click', () => aturHuruf(tombolHuruf.getAttribute('aria-pressed') !== 'true'));

tampilkanPercakapan();

// Pertanyaan dari halaman depan (/tanya?q=...) dikirim sekali di percakapan baru,
// lalu dihapus dari alamat agar tidak terkirim lagi saat halaman dimuat ulang.
const pertanyaanAwal = (new URLSearchParams(location.search).get('q') || '').trim().slice(0, 500);
if (pertanyaanAwal) {
  history.replaceState(null, '', location.pathname + (modeAwal === 'llm' ? '?mode=llm' : ''));
  if (aktif.pesan.length) { aktif = percakapanBaru(aktif.mode); data.aktif = null; tampilkanPercakapan(); }
  kirim(pertanyaanAwal);
}
