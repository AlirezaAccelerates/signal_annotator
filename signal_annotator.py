#!/usr/bin/env python3
"""Local WAV quality annotation. Dependencies: numpy, scipy; UI: local web browser."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from scipy.io import wavfile


def discover(paths):
    files = set()
    for item in paths:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == '.wav':
            files.add(path)
        elif path.is_dir():
            files.update(p.resolve() for p in path.rglob('*')
                         if p.is_file() and p.suffix.lower() == '.wav')
        else:
            raise ValueError(f'Not a WAV file or directory: {path}')
    return sorted(files)


def window_grid(frames, sr, seconds, step):
    if not all(math.isfinite(v) and v > 0 for v in (seconds, step)):
        raise ValueError('Window and step must be finite, positive seconds.')
    width, hop = round(seconds * sr), round(step * sr)
    if width < 1 or hop < 1:
        raise ValueError('Window and step must each be at least one sample.')
    if hop > width:
        raise ValueError('Step cannot exceed window size (this would leave gaps).')
    return width, hop, range(0, frames, hop)


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


class Annotator:
    def __init__(self):
        self.signal = None
        self.files = []
        self.revision = 0

    def load(self, paths, output, window, step):
        self.save()
        seconds, step = float(window), float(step)
        window_grid(1, 48000, seconds, step)
        files = discover(paths)
        if not files:
            raise ValueError('No WAV files found. Enter one folder or WAV path per line.')
        if not output.strip():
            raise ValueError('Enter an output directory.')
        out = Path(output).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        self.open_file(0, files, out, seconds, step)
        self.revision += 1

    def open_file(self, position, files=None, out=None, seconds=None, step=None):
        files = self.files if files is None else files
        out = self.out if out is None else out
        seconds = self.seconds if seconds is None else seconds
        step = self.step_seconds if step is None else step
        path = files[position]
        sr, signal = wavfile.read(str(path))
        if signal.ndim not in (1, 2) or not len(signal):
            raise ValueError(f'Empty or unsupported WAV: {path}')
        if np.issubdtype(signal.dtype, np.floating) and not np.isfinite(signal).all():
            raise ValueError(f'WAV contains nonfinite samples: {path}')
        width, hop, starts = window_grid(len(signal), sr, seconds, step)
        stat = path.stat()
        metadata = dict(source=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                        sample_rate=int(sr), frames=len(signal), window_samples=width,
                        step_samples=hop, window_seconds=seconds, step_seconds=step)
        identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
        base = out / f'{path.stem}.{identity}'
        state_path = base.with_suffix(base.suffix + '.progress.json')
        text_path = base.with_suffix(base.suffix + '.txt')
        labels = [None] * len(starts)
        index = 0
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            saved = state.get('labels')
            if state.get('metadata') != metadata or not isinstance(saved, list) or len(saved) != len(labels) or any(v is not None and (type(v) is not int or v not in (0, 1)) for v in saved):
                raise ValueError(f'Invalid saved progress: {state_path}')
            labels = saved
            index = labels.index(None) if None in labels else min(max(int(state.get('index', 0)), 0), len(labels)-1)
        self.files, self.out = files, out
        self.seconds, self.step_seconds = seconds, step
        self.sr, self.signal = sr, signal
        self.width, self.hop, self.starts = width, hop, starts
        self.metadata, self.state_path, self.text_path = metadata, state_path, text_path
        self.labels, self.index, self.file_index = labels, index, position
        # Fixed full-file scale prevents quiet windows looking artificially loud.
        values = signal.reshape(len(signal), -1)
        self.center = 128 if signal.dtype == np.uint8 else 0
        self.scale = np.maximum(np.maximum(np.abs(values.min(axis=0).astype(float)-self.center),
                                          np.abs(values.max(axis=0).astype(float)-self.center)), 1e-12)


    def save(self):
        if self.signal is None:
            return
        atomic_write(self.state_path, json.dumps(dict(metadata=self.metadata, labels=self.labels,
                                                     index=self.index), indent=2) + '\n')
        if None not in self.labels:
            atomic_write(self.text_path, ','.join(map(str, self.labels)) + '\n')


    def act(self, action, value, revision):
        if self.signal is None:
            raise ValueError('Load WAV files first.')
        if revision != self.revision:
            raise ValueError('The window changed. Refresh before annotating.')
        if action == 'label':
            if type(value) is not int or value not in (0, 1):
                raise ValueError('Label must be 0 or 1.')
            self.labels[self.index] = value
            self.save()
        elif action == 'move':
            if value not in (-1, 1):
                raise ValueError('Invalid direction.')
            if self.labels[self.index] is None:
                raise ValueError('Choose Good or Poor before leaving this window.')
            self.save()
            destination = self.index + value
            if 0 <= destination < len(self.labels):
                self.index = destination
            elif value > 0 and self.file_index + 1 < len(self.files):
                self.open_file(self.file_index + 1)
            elif value < 0 and self.file_index > 0:
                self.open_file(self.file_index - 1)
                self.index = len(self.labels) - 1
        elif action == 'save':
            self.save()
        else:
            raise ValueError('Unknown action.')
        self.revision += 1

    def view(self):
        if self.signal is None:
            return dict(loaded=False, revision=self.revision)
        start = self.starts[self.index]
        end = min(start + self.width, len(self.signal))
        data = self.signal[start:end].reshape(end-start, -1)
        # Only a peak-preserving display envelope is sent to the browser.
        edges = np.linspace(0, len(data), min(1400, len(data))+1, dtype=int)
        channels = []
        for channel in range(data.shape[1]):
            values = (data[:, channel].astype(float)-self.center)/self.scale[channel]
            channels.append(dict(low=np.minimum.reduceat(values, edges[:-1]).tolist(),
                                 high=np.maximum.reduceat(values, edges[:-1]).tolist()))
        complete = None not in self.labels
        return dict(loaded=True, revision=self.revision, name=self.files[self.file_index].name,
                    source=str(self.files[self.file_index]), file=self.file_index+1, files=len(self.files),
                    window=self.index+1, windows=len(self.labels), label=self.labels[self.index],
                    labeled=sum(v is not None for v in self.labels), sr=int(self.sr),
                    start=start/self.sr, end=end/self.sr, channels=channels, complete=complete,
                    first=self.file_index == 0 and self.index == 0,
                    last=self.file_index == len(self.files)-1 and self.index == len(self.labels)-1,
                    result=str(self.text_path if complete else self.state_path))

PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Signal quality annotation</title>
<style>
*{box-sizing:border-box}body{font:16px system-ui,sans-serif;background:#eef2f6;color:#17243b;margin:0;padding:28px}
main{max-width:1150px;margin:auto}h1{margin:0 0 8px}p{color:#526176}section{background:white;border:1px solid #dce3ed;border-radius:12px;padding:20px;margin:20px 0}
label{display:block;font-weight:600;margin-bottom:8px}input,textarea,button{font:inherit;border:1px solid #b8c4d4;border-radius:6px;padding:10px}input,textarea{width:100%}textarea{height:85px}button{cursor:pointer;background:#fff}button:hover{background:#edf3fc}button:disabled{opacity:.45;cursor:default}.row{display:flex;gap:16px;align-items:end;flex-wrap:wrap}.field{flex:1;min-width:150px}.primary{background:#245be0;color:white}.selected{background:#173f9d;color:white;outline:3px solid #b3c8ff}#plot{width:100%;height:360px;background:#f8fafc}#status{overflow-wrap:anywhere}#error{color:#a22222;min-height:24px}#meta{color:#526176}#good,#poor{min-width:140px}
</style>
<main><h1>Signal quality annotation</h1><p>Review each time window and mark its signal quality. Labels save automatically.</p>
<section><label for="paths">WAV folders or files — one full path per line</label><textarea id="paths" placeholder="/path/to/recordings"></textarea>
<label for="output">Output directory</label><input id="output" placeholder="/path/to/annotations">
<div class="row" style="margin-top:16px"><div class="field"><label for="seconds">Window (seconds)</label><input id="seconds" type="number" min="0.000001" step="any"></div><div class="field"><label for="step">Step (seconds)</label><input id="step" type="number" min="0.000001" step="any"></div><button id="load" class="primary">Load / apply settings</button></div></section>
<section><h2 id="title">No WAV loaded</h2><p id="meta">Enter input paths and an output directory above.</p><canvas id="plot" aria-label="Waveform over time"></canvas>
<div class="row" style="margin-top:18px"><button id="back">← Previous</button><button id="good" aria-pressed="false">Good (0)</button><button id="poor" aria-pressed="false">Poor (1)</button><button id="next">Next →</button><button id="save">Save progress</button></div>
<p id="status" role="status"></p><div id="error" role="alert"></div><small>Keys: 0 = good · 1 = poor · ← / → = navigate. Select a label before leaving a window.</small></section></main>
<script>
const defaults=__DEFAULTS__, token=__TOKEN__;
const el=id=>document.getElementById(id);
let state={loaded:false},busy=false;
el('paths').value=defaults.paths.join('\n');el('output').value=defaults.output;el('seconds').value=defaults.window;el('step').value=defaults.step;
function controls(){for(const id of ['load','back','next','good','poor','save'])el(id).disabled=busy||(id!=='load'&&!state.loaded);if(state.loaded)el('back').disabled=busy||state.first;}
async function request(action,extra={}){if(busy)return;busy=true;controls();el('error').textContent='';try{const response=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json','X-Annotation-Token':token},body:JSON.stringify({action,revision:state.revision,...extra})});const data=await response.json();if(!response.ok)throw Error(data.error);state=data;render();}catch(error){el('error').textContent=error.message;}finally{busy=false;controls();}}
function render(){if(!state.loaded)return;el('title').textContent=`${state.name} · File ${state.file}/${state.files} · Window ${state.window}/${state.windows}`;el('meta').textContent=`${state.start.toFixed(3)}–${state.end.toFixed(3)} seconds · ${state.sr.toLocaleString()} Hz · ${state.labeled}/${state.windows} labeled`;
for(const [id,value] of [['good',0],['poor',1]]){el(id).classList.toggle('selected',state.label===value);el(id).setAttribute('aria-pressed',state.label===value);}
el('next').textContent=state.last?'Finish':state.window===state.windows?'Next file →':'Next →';el('status').textContent=(state.complete?'Completed text file: ':'Progress file: ')+state.result;draw();}
function draw(){const canvas=el('plot'),dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=Math.max(300,(state.channels?.length||1)*130);canvas.style.height=h+'px';canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);if(!state.loaded)return;const left=58,right=w-24,top=18,bottom=h-44;
ctx.font='12px system-ui';ctx.textAlign='center';for(let i=0;i<=5;i++){const x=left+(right-left)*i/5;ctx.strokeStyle='#e0e6ee';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.fillStyle='#526176';ctx.fillText((state.start+(state.end-state.start)*i/5).toFixed(3),x,bottom+20);}ctx.fillText('Time (seconds)',(left+right)/2,h-4);
state.channels.forEach((channel,c)=>{const lane=(bottom-top)/state.channels.length,mid=top+lane*(c+.5),amp=lane*.42;ctx.fillStyle='#526176';ctx.fillText('Ch '+(c+1),25,mid);ctx.strokeStyle='#aab6c5';ctx.beginPath();ctx.moveTo(left,mid);ctx.lineTo(right,mid);ctx.stroke();ctx.strokeStyle='#245be0';ctx.beginPath();for(let i=0;i<channel.low.length;i++){const x=left+(i+.5)*(right-left)/channel.low.length,y=mid-channel.high[i]*amp;ctx.moveTo(x,y);ctx.lineTo(x,Math.max(y+1,mid-channel.low[i]*amp));}ctx.stroke();});}
el('load').onclick=()=>request('load',{paths:el('paths').value.split('\n').map(s=>s.trim()).filter(Boolean),output:el('output').value,window:Number(el('seconds').value),step:Number(el('step').value)});
el('good').onclick=()=>request('label',{value:0});el('poor').onclick=()=>request('label',{value:1});el('back').onclick=()=>request('move',{value:-1});el('next').onclick=async()=>{await request('move',{value:1});if(state.last&&state.complete&&!el('error').textContent)el('status').textContent='All files complete. Saved: '+state.result;};el('save').onclick=()=>request('save');
window.addEventListener('resize',draw);document.addEventListener('keydown',event=>{if(['INPUT','TEXTAREA'].includes(event.target.tagName)||event.ctrlKey||event.metaKey||event.altKey||!state.loaded||busy)return;const keys={'0':'good','1':'poor',ArrowLeft:'back',ArrowRight:'next'};if(keys[event.key]){event.preventDefault();el(keys[event.key]).click();}});request('state');
</script></html>'''


def make_handler(app, defaults, token):
    page = PAGE.replace('__DEFAULTS__', json.dumps(defaults).replace('<', '\\u003c')).replace('__TOKEN__', json.dumps(token)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data, content_type='application/json'):
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(body)

        def allowed_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.allowed_host():
                return self.reply(403, dict(error='Invalid host'))
            if self.path != '/':
                return self.reply(404, dict(error='Not found'))
            self.reply(200, page, 'text/html')

        def do_POST(self):
            if not self.allowed_host() or self.headers.get('X-Annotation-Token') != token:
                return self.reply(403, dict(error='Invalid session'))
            if self.path != '/api':
                return self.reply(404, dict(error='Not found'))
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 100000:
                    raise ValueError('Invalid request size.')
                data = json.loads(self.rfile.read(size))
                action = data['action']
                if action == 'load':
                    app.load(data['paths'], data['output'], data['window'], data['step'])
                elif action != 'state':
                    app.act(action, data.get('value'), data.get('revision'))
                self.reply(200, app.view())
            except Exception as exc:
                self.reply(400, dict(error=str(exc)))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='WAV directories or files')
    parser.add_argument('-o', '--output', default='', help='Output directory')
    parser.add_argument('--window', type=float, default=5, help='Window seconds')
    parser.add_argument('--step', type=float, default=5, help='Step seconds')
    parser.add_argument('--port', type=int, default=0, help='Local port (default: choose available port)')
    parser.add_argument('--no-browser', action='store_true', help='Print URL without opening browser')
    args = parser.parse_args()
    app = Annotator()
    if args.paths and args.output:
        try:
            app.load(args.paths, args.output, args.window, args.step)
        except Exception as exc:
            print(f'Initial load failed: {exc}\nCorrect the settings in the browser.')
    defaults = dict(paths=args.paths, output=args.output, window=args.window, step=args.step)
    server = HTTPServer(('127.0.0.1', args.port), make_handler(app, defaults, secrets.token_urlsafe(32)))
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'Open {url}\nKeep this terminal open. Press Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            app.save()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
