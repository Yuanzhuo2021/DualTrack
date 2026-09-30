(() => {
  'use strict';
  const data = window.DUALTRACK_SAMPLES.comparison;
  const pageSize = 6;
  const $ = id => document.getElementById(id);
  const languages = {english: 'ENGLISH', spanish: 'SPANISH', chinese: 'CHINESE', japanese: 'JAPANESE'};
  const sourceOrder = ['GT', 'GELINA', 'DualTrack'];
  let page = 0;

  const element = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function render() {
    document.querySelectorAll('video').forEach(video => video.pause());
    const query = $('search').value.trim().toLowerCase();
    const language = $('language').value;
    const model = $('model').value;
    const rows = data.filter(row =>
      (language === 'all' || row.language === language) &&
      (model === 'all' || row.models[model]) &&
      (!query || (row.id + ' ' + row.text).toLowerCase().includes(query))
    );
    const pages = Math.max(1, Math.ceil(rows.length / pageSize));
    page = Math.min(page, pages - 1);
    const totalVideos = rows.reduce((total, row) => total + (model === 'all' ? Object.keys(row.models).length : 1), 0);
    $('results').textContent = `${rows.length} samples · ${totalVideos} videos`;
    $('gallery').replaceChildren();
    if (!rows.length) {
      $('gallery').append(element('p', 'empty', 'No samples match these filters. Try another language, source, or search.'));
    }
    rows.slice(page * pageSize, (page + 1) * pageSize).forEach(row => {
      const section = element('section', 'sample');
      const heading = element('div', 'sample-header');
      const id = element('div', 'sample-id');
      id.append(element('span', 'badge', languages[row.language]), element('span', '', row.id));
      heading.append(id, element('p', 'sample-text', row.text));
      section.append(heading);
      const names = sourceOrder.filter(name => row.models[name] && (model === 'all' || name === model));
      const players = element('div', 'players' + (names.length === 1 ? ' single' : names.length === 3 ? ' three' : ''));
      names.forEach(name => {
        const asset = row.models[name];
        const card = element('article', 'player');
        const header = element('div', 'player-heading');
        header.append(element('strong', '', name === 'GT' ? 'GT (ground truth)' : name), element('span', '', asset.duration.toFixed(2) + ' s'));
        const video = element('video');
        video.controls = true;
        video.playsInline = true;
        video.preload = 'metadata';
        video.src = asset.video;
        video.poster = asset.poster;
        video.setAttribute('aria-label', `${name} · ${row.language} · ${row.id}`);
        const links = element('div', 'player-links');
        const open = element('a', '', 'Open video ↗');
        open.href = asset.video;
        links.append(open);
        card.append(header, video, links);
        players.append(card);
      });
      section.append(players);
      $('gallery').append(section);
    });
    $('page').textContent = `Page ${page + 1} of ${pages}`;
    $('previous').disabled = page === 0;
    $('next').disabled = page >= pages - 1;
  }

  ['language', 'model'].forEach(id => $(id).addEventListener('change', () => {page = 0; render();}));
  $('search').addEventListener('input', () => {page = 0; render();});
  $('reset').addEventListener('click', () => {
    $('language').value = 'all';
    $('model').value = 'all';
    $('search').value = '';
    page = 0;
    render();
  });
  $('previous').addEventListener('click', () => {page--; render(); $('results').scrollIntoView({block: 'start'});});
  $('next').addEventListener('click', () => {page++; render(); $('results').scrollIntoView({block: 'start'});});
  document.addEventListener('play', event => {
    if (event.target.tagName === 'VIDEO') {
      document.querySelectorAll('video').forEach(video => {if (video !== event.target) video.pause();});
    }
  }, true);
  render();
})();
