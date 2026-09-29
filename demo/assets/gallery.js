(() => {
  'use strict';
  const data = window.DUALTRACK_SAMPLES, pageSize = 6;
  const $ = id => document.getElementById(id);
  let mode = 'comparison', page = 0;
  const languages = {english:'ENGLISH',spanish:'SPANISH',chinese:'CHINESE',japanese:'JAPANESE'};
  const element = (tag, cls, text) => {const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node;};
  const link = (text, href) => {const a=element('a','',text);a.href=href;return a;};
  function render() {
    document.querySelectorAll('video').forEach(v=>v.pause());
    const query=$('search').value.trim().toLowerCase(), language=$('language').value, model=$('model').value;
    const rows=data[mode].filter(r=>(language==='all'||r.language===language)&&(model==='all'||r.models[model])&&(!query||(r.id+' '+r.text).toLowerCase().includes(query)));
    const pages=Math.max(1,Math.ceil(rows.length/pageSize));page=Math.min(page,pages-1);
    const totalVideos=rows.reduce((n,r)=>n+(model==='all'?Object.keys(r.models).length:1),0);
    $('results').textContent=`${rows.length} samples · ${totalVideos} videos`;
    $('collection-note').textContent=mode==='comparison'?'DualTrack and GELINA on matched English/Spanish inputs, plus DualTrack Chinese/Japanese examples. Original 960 × 960 renders with corresponding generated audio.':'Original study media: ground truth, GELINA, and DualTrack for 28 matched English/Spanish inputs. Neutral 720 × 720 renders, shown with source labels.';
    $('gallery').replaceChildren();
    if(!rows.length)$('gallery').append(element('p','empty','No samples match these filters. Try another language, model, or search.'));
    rows.slice(page*pageSize,(page+1)*pageSize).forEach(r=>{
      const section=element('section','sample'), heading=element('div','sample-header'), id=element('div','sample-id');
      id.append(element('span','badge',languages[r.language]),element('span','',r.id));heading.append(id,element('p','sample-text',r.text));section.append(heading);
      const order=mode==='comparison'?['GELINA','DualTrack']:['Ground truth','GELINA','DualTrack'];
      const names=order.filter(name=>r.models[name]&&(model==='all'||name===model));
      const players=element('div','players'+(names.length===1?' single':names.length===3?' three':''));
      names.forEach(name=>{
        const asset=r.models[name], card=element('article','player'), header=element('div','player-heading');
        header.append(element('strong','',name),element('span','',asset.duration?asset.duration.toFixed(2)+' s':'Audio + motion'));
        const video=element('video');video.controls=true;video.playsInline=true;video.preload='none';video.src=asset.video;video.setAttribute('aria-label',`${name} · ${r.language} · ${r.id}`);if(asset.poster)video.poster=asset.poster;
        const links=element('div','player-links');links.append(link('Open video ↗',asset.video));if(asset.motion)links.append(link('Motion only',asset.motion));if(asset.speech)links.append(link('Speech only',asset.speech));
        card.append(header,video,links);players.append(card);
      });section.append(players);$('gallery').append(section);
    });
    $('page').textContent=`Page ${page+1} of ${pages}`;$('previous').disabled=page===0;$('next').disabled=page>=pages-1;
  }
  ['language','model'].forEach(id=>$(id).addEventListener('change',()=>{page=0;render();}));
  $('search').addEventListener('input',()=>{page=0;render();});
  $('reset').addEventListener('click',()=>{$('language').value='all';$('model').value='all';$('search').value='';page=0;render();});
  document.querySelectorAll('[data-mode]').forEach(tab=>tab.addEventListener('click',()=>{mode=tab.dataset.mode;page=0;$('model').value='all';$('language').value='all';$('search').value='';document.querySelectorAll('[data-mode]').forEach(t=>{const active=t===tab;t.classList.toggle('active',active);t.setAttribute('aria-pressed',String(active));});render();}));
  $('previous').addEventListener('click',()=>{page--;render();$('results').scrollIntoView({block:'start'});});
  $('next').addEventListener('click',()=>{page++;render();$('results').scrollIntoView({block:'start'});});
  document.addEventListener('play',event=>{if(event.target.tagName==='VIDEO')document.querySelectorAll('video').forEach(v=>{if(v!==event.target)v.pause();});},true);
  render();
})();
