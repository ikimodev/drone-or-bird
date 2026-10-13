import os
import io
import time
import json
import base64
import urllib.parse
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
from PIL import Image

from defense_core import AirDefenseEnsemble
from hud_renderer import render_tactical_hud

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

# Initialize defense ensemble
defense_engine = AirDefenseEnsemble()

HTML_PAGE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>منظومة الدفاع الجوي والرصد الذكي | C2 Anti-Drone & Avian Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #050811;
            --bg-panel: rgba(10, 16, 30, 0.90);
            --border-panel: rgba(0, 210, 255, 0.18);
            --border-highlight: rgba(0, 210, 255, 0.45);
            --color-drone: #ff1744;
            --color-drone-glow: rgba(255, 23, 68, 0.35);
            --color-bird: #00e676;
            --color-bird-glow: rgba(0, 230, 118, 0.35);
            --color-cyan: #00d2ff;
            --text-main: #f0f6fc;
            --text-dim: #7d8da6;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Cairo', system-ui, sans-serif;
        }

        body {
            background-color: var(--bg-deep);
            background-image: 
                radial-gradient(at 0% 0%, rgba(0, 210, 255, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(255, 23, 68, 0.08) 0px, transparent 50%),
                linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
            color: var(--text-main);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1460px;
            margin: 0 auto;
        }

        /* Tactical Header */
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-panel);
            border: 1px solid var(--border-panel);
            border-radius: 16px;
            padding: 16px 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            flex-wrap: wrap;
            gap: 16px;
        }

        .logo-wrap {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .radar-badge {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: radial-gradient(circle, rgba(0, 210, 255, 0.25) 0%, rgba(0,0,0,0.8) 100%);
            border: 1px solid var(--color-cyan);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            box-shadow: 0 0 16px rgba(0, 210, 255, 0.3);
            position: relative;
            overflow: hidden;
        }

        .radar-badge::after {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border-top: 2px solid var(--color-cyan);
            border-radius: 50%;
            animation: sweep 2s linear infinite;
        }

        @keyframes sweep {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        h1 {
            font-size: 22px;
            font-weight: 900;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff, var(--color-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-sub {
            font-size: 13px;
            color: var(--text-dim);
            font-family: var(--font-mono);
        }

        .header-telemetry {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .alert-indicator {
            padding: 8px 18px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            font-family: var(--font-mono);
            display: flex;
            align-items: center;
            gap: 10px;
            border: 1px solid rgba(0, 230, 118, 0.4);
            background: rgba(0, 230, 118, 0.12);
            color: var(--color-bird);
            transition: all 0.3s ease;
        }

        .alert-indicator.drone-alert {
            border-color: var(--color-drone);
            background: rgba(255, 23, 68, 0.2);
            color: var(--color-drone);
            box-shadow: 0 0 20px var(--color-drone-glow);
            animation: blinkAlert 1s infinite alternate;
        }

        @keyframes blinkAlert {
            0% { opacity: 0.8; }
            100% { opacity: 1; transform: scale(1.02); }
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 8px currentColor;
        }

        /* Top Command Toolbar */
        .toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            gap: 14px;
            flex-wrap: wrap;
        }

        .nav-tabs {
            display: flex;
            gap: 8px;
            background: var(--bg-panel);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--border-panel);
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-dim);
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .tab-btn.active {
            background: rgba(0, 210, 255, 0.16);
            color: #ffffff;
            border: 1px solid var(--color-cyan);
            box-shadow: 0 0 16px rgba(0, 210, 255, 0.2);
        }

        .sound-toggle-btn {
            background: var(--bg-panel);
            border: 1px solid var(--border-panel);
            color: var(--text-main);
            padding: 10px 18px;
            border-radius: 12px;
            font-size: 13.5px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }

        .sound-toggle-btn.sound-on {
            border-color: var(--color-cyan);
            color: var(--color-cyan);
        }

        /* Main Workspace Grid */
        .grid-layout {
            display: grid;
            grid-template-columns: 2fr 1.1fr;
            gap: 20px;
        }

        @media (max-width: 1080px) {
            .grid-layout {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-panel);
            border: 1px solid var(--border-panel);
            border-radius: 18px;
            padding: 20px;
            backdrop-filter: blur(16px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        .card-title {
            font-size: 17px;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: -0.2px;
        }

        /* Viewport Screen */
        .viewport-screen {
            width: 100%;
            height: 520px;
            background: #020409;
            border-radius: 14px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            border: 2px dashed rgba(0, 210, 255, 0.2);
            transition: all 0.2s ease;
        }

        .viewport-screen.dragover {
            border-color: var(--color-cyan);
            background: rgba(0, 210, 255, 0.05);
        }

        .viewport-screen img, .viewport-screen video {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .drop-prompt {
            text-align: center;
            padding: 20px;
            cursor: pointer;
        }

        .drop-prompt span {
            font-size: 56px;
            display: block;
            margin-bottom: 12px;
        }

        .drop-prompt h3 {
            font-size: 19px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .drop-prompt p {
            font-size: 13px;
            color: var(--text-dim);
            margin-bottom: 16px;
        }

        .controls-row {
            display: flex;
            gap: 10px;
            margin-top: 16px;
            flex-wrap: wrap;
        }

        .btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-panel);
            color: var(--text-main);
            padding: 9px 16px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn:hover {
            background: rgba(0, 210, 255, 0.15);
            border-color: var(--color-cyan);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .btn-cyan {
            background: rgba(0, 210, 255, 0.18);
            border-color: var(--color-cyan);
            color: #ffffff;
        }

        .btn-red {
            background: rgba(255, 23, 68, 0.18);
            border-color: var(--color-drone);
            color: #ffffff;
        }
        .btn-red:hover {
            background: rgba(255, 23, 68, 0.35);
            border-color: #ff5252;
        }

        .btn-green {
            background: rgba(0, 230, 118, 0.18);
            border-color: var(--color-bird);
            color: #ffffff;
        }
        .btn-green:hover {
            background: rgba(0, 230, 118, 0.35);
            border-color: #69f0ae;
        }

        input[type="file"] {
            display: none;
        }

        /* Stats & Telemetry Section */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }

        .stat-metric {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 14px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .stat-metric.metric-drone {
            border-color: rgba(255, 23, 68, 0.3);
            background: rgba(255, 23, 68, 0.05);
        }

        .stat-metric.metric-bird {
            border-color: rgba(0, 230, 118, 0.3);
            background: rgba(0, 230, 118, 0.05);
        }

        .stat-metric-title {
            font-size: 13px;
            color: var(--text-dim);
            font-weight: 700;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .stat-metric-val {
            font-size: 32px;
            font-weight: 900;
            font-family: var(--font-mono);
        }

        .metric-drone .stat-metric-val {
            color: var(--color-drone);
            text-shadow: 0 0 16px var(--color-drone-glow);
        }

        .metric-bird .stat-metric-val {
            color: var(--color-bird);
            text-shadow: 0 0 16px var(--color-bird-glow);
        }

        /* Radar Box */
        .radar-box {
            width: 100%;
            height: 190px;
            background: #020610;
            border-radius: 14px;
            border: 1px solid var(--border-panel);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }

        .radar-circle {
            width: 160px;
            height: 160px;
            border-radius: 50%;
            border: 1px solid rgba(0, 210, 255, 0.3);
            position: relative;
            box-shadow: inset 0 0 20px rgba(0, 210, 255, 0.1);
        }

        .radar-circle::before {
            content: '';
            position: absolute;
            width: 100px;
            height: 100px;
            top: 30px;
            left: 30px;
            border-radius: 50%;
            border: 1px dashed rgba(0, 210, 255, 0.2);
        }

        .radar-crosshair-h, .radar-crosshair-v {
            position: absolute;
            background: rgba(0, 210, 255, 0.2);
        }
        .radar-crosshair-h {
            width: 100%;
            height: 1px;
            top: 50%;
            left: 0;
        }
        .radar-crosshair-v {
            width: 1px;
            height: 100%;
            left: 50%;
            top: 0;
        }

        .radar-sweep-arm {
            position: absolute;
            width: 80px;
            height: 80px;
            top: 0;
            left: 80px;
            transform-origin: 0 80px;
            background: linear-gradient(45deg, rgba(0, 210, 255, 0.4) 0%, transparent 80%);
            border-radius: 0 100% 0 0;
            animation: radarRotate 3s linear infinite;
        }

        @keyframes radarRotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .blip-drone {
            position: absolute;
            width: 8px;
            height: 8px;
            background: var(--color-drone);
            border-radius: 50%;
            top: 45px;
            left: 110px;
            box-shadow: 0 0 10px var(--color-drone);
            animation: ping 1.2s infinite;
        }

        .blip-bird {
            position: absolute;
            width: 8px;
            height: 8px;
            background: var(--color-bird);
            border-radius: 50%;
            top: 115px;
            left: 50px;
            box-shadow: 0 0 10px var(--color-bird);
            animation: ping 1.5s infinite;
        }

        @keyframes ping {
            0% { transform: scale(0.8); opacity: 0.5; }
            50% { transform: scale(1.4); opacity: 1; }
            100% { transform: scale(0.8); opacity: 0.5; }
        }

        /* Forensics Target Cards */
        .forensics-container {
            max-height: 290px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .forensics-container::-webkit-scrollbar {
            width: 6px;
        }
        .forensics-container::-webkit-scrollbar-thumb {
            background: rgba(0, 210, 255, 0.2);
            border-radius: 4px;
        }

        .forensic-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
            display: flex;
            gap: 12px;
            align-items: center;
            transition: all 0.2s;
        }

        .forensic-card:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: var(--border-highlight);
        }

        .forensic-card.card-drone {
            border-color: rgba(255, 23, 68, 0.35);
            background: rgba(255, 23, 68, 0.04);
        }

        .forensic-card.card-bird {
            border-color: rgba(0, 230, 118, 0.35);
            background: rgba(0, 230, 118, 0.04);
        }

        .crop-thumb {
            width: 64px;
            height: 64px;
            border-radius: 8px;
            object-fit: cover;
            background: #000;
            border: 1px solid var(--border-panel);
            flex-shrink: 0;
        }

        .card-drone .crop-thumb {
            border-color: var(--color-drone);
            box-shadow: 0 0 8px var(--color-drone-glow);
        }

        .card-bird .crop-thumb {
            border-color: var(--color-bird);
            box-shadow: 0 0 8px var(--color-bird-glow);
        }

        .forensic-info {
            flex-grow: 1;
            font-size: 13px;
        }

        .forensic-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }

        .badge-type {
            font-size: 11.5px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 6px;
            font-family: var(--font-mono);
        }

        .badge-drone {
            background: rgba(255, 23, 68, 0.25);
            color: #ff5252;
            border: 1px solid var(--color-drone);
        }

        .badge-bird {
            background: rgba(0, 230, 118, 0.25);
            color: #69f0ae;
            border: 1px solid var(--color-bird);
        }

        .forensic-sub {
            font-size: 12px;
            color: #cbd5e1;
            font-weight: 700;
            margin-bottom: 2px;
        }

        .forensic-meta {
            font-size: 11px;
            color: var(--text-dim);
            font-family: var(--font-mono);
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        /* Loading spinner */
        .spinner {
            width: 48px;
            height: 48px;
            border: 4px solid rgba(0, 210, 255, 0.15);
            border-top: 4px solid var(--color-cyan);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        #loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(2, 4, 9, 0.82);
            backdrop-filter: blur(6px);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Tactical Header -->
        <header>
            <div class="logo-wrap">
                <div class="radar-badge">🛰️</div>
                <div>
                    <h1>منظومة الدفاع الجوي والرصد الذكي</h1>
                    <div class="header-sub">C2 AI AIRSPACE SURVEILLANCE & TARGET DISCRIMINATION</div>
                </div>
            </div>

            <div class="header-telemetry">
                <div class="alert-indicator" id="airspace-alert">
                    <div class="status-dot"></div>
                    <span id="airspace-alert-text">جاري فحص الأجواء...</span>
                </div>
            </div>
        </header>

        <!-- Command Toolbar -->
        <div class="toolbar">
            <div class="nav-tabs">
                <button class="tab-btn active" id="tab-image-btn" onclick="switchTab('image')">
                    <span>🛰️</span> تحليل الصور والأهداف البصرية
                </button>
                <button class="tab-btn" id="tab-video-btn" onclick="switchTab('video')">
                    <span>📹</span> بث المراقبة الحية والرادار (Stream)
                </button>
            </div>

            <button class="sound-toggle-btn" id="sound-btn" onclick="toggleAudio()">
                <span id="sound-icon">🔇</span>
                <span id="sound-text">تفعيل صفارة الإنذار التكتيكية</span>
            </button>
        </div>

        <!-- Main Workspace -->
        <div class="grid-layout">
            <!-- Left Screen Viewport -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span>🎯</span> شاشة الرصد التكتيكي والرؤية البصرية (HUD)
                    </div>
                    <div style="font-size: 12px; color: var(--color-cyan); font-family: var(--font-mono);">
                        ENSEMBLE: SHAHED-v8s + MULTICOPTER + AVIAN-v8m
                    </div>
                </div>

                <!-- Image Mode Panel -->
                <div id="image-panel">
                    <div class="viewport-screen" id="image-dropzone" onclick="document.getElementById('image-input').click()">
                        <div id="loading-overlay" style="display: none;">
                            <div class="spinner"></div>
                            <div style="font-weight: 700; color: var(--color-cyan);">جاري المسح الطيفي والتحليل البصري...</div>
                        </div>

                        <div class="drop-prompt" id="drop-prompt">
                            <span>🛰️</span>
                            <h3>اسحب وأفلت أي صورة هنا أو اضغط للاختيار</h3>
                            <p>يدعم جميع الصيغ (AVIF, WEBP, PNG, JPG) بالإضافة للصق المباشر عبر Ctrl+V</p>
                            <button class="btn btn-cyan" onclick="event.stopPropagation(); document.getElementById('image-input').click()">📁 استعراض صورة من جهازك</button>
                        </div>
                        <img id="processed-image" style="display: none;" alt="Tactical HUD Output">
                    </div>

                    <input type="file" id="image-input" accept="image/*,.avif,.webp,.png,.jpg,.jpeg" onchange="handleFileSelect(event)">

                    <div class="controls-row">
                        <button class="btn btn-red" onclick="testSample('user_image.jpg')">🛸 درون انتحاري (Shahed-136)</button>
                        <button class="btn btn-green" onclick="testSample('soaring_birds.jpg')">🦅 طيور محلقة في السماء (Soaring)</button>
                        <button class="btn btn-red" onclick="testSample('sample_drone.jpg')">🚁 درون مروحي رباعي (Multicopter)</button>
                        <button class="btn btn-green" onclick="testSample('sample_bird.jpg')">🐣 عش طيور صغيرة (Avian Nest)</button>
                    </div>
                </div>

                <!-- Video Surveillance Mode -->
                <div id="video-panel" style="display: none;">
                    <div class="viewport-screen">
                        <img id="mjpeg-stream" src="" alt="Live Video Feed">
                    </div>
                    <div class="controls-row">
                        <button class="btn btn-cyan" onclick="restartStream()">🔄 إعادة مزامنة البث</button>
                        <a href="/output_test.mp4" download="air_defense_recorded.mp4" class="btn">⬇️ تنزيل تسجيل الرادار (MP4)</a>
                    </div>
                </div>
            </div>

            <!-- Right Telemetry & Optical Forensics Pod -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span>📡</span> شاشة الرادار والاستطلاع البصري
                    </div>
                </div>

                <!-- Tactical Radar Scope -->
                <div class="radar-box">
                    <div class="radar-circle">
                        <div class="radar-crosshair-h"></div>
                        <div class="radar-crosshair-v"></div>
                        <div class="radar-sweep-arm"></div>
                        <div class="blip-drone" id="radar-blip-drone" style="display: none;"></div>
                        <div class="blip-bird" id="radar-blip-bird" style="display: none;"></div>
                    </div>
                </div>

                <!-- Key Metrics Grid -->
                <div class="stats-grid">
                    <div class="stat-metric metric-drone">
                        <div class="stat-metric-title">
                            <span>🚨</span> أهداف الدرون (UAS)
                        </div>
                        <div class="stat-metric-val" id="metric-drone-val">0</div>
                    </div>
                    <div class="stat-metric metric-bird">
                        <div class="stat-metric-title">
                            <span>🐦</span> كائنات حية (Avian)
                        </div>
                        <div class="stat-metric-val" id="metric-bird-val">0</div>
                    </div>
                </div>

                <!-- Optical Forensics Target Crops -->
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: var(--color-cyan); margin-bottom: 8px; display: flex; justify-content: space-between;">
                        <span>🔬 عدسة الاستطلاع والتحليل البصري:</span>
                        <span id="target-count-badge" style="font-family: var(--font-mono); color: var(--text-dim);">0 أهداف</span>
                    </div>
                    <div class="forensics-container" id="forensics-list">
                        <div style="color: var(--text-dim); text-align: center; padding: 24px; font-size: 13px;">
                            الرادار جاهز ومستعد لرصد الأهداف...
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let audioEnabled = false;
        let audioCtx = null;

        function toggleAudio() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            audioEnabled = !audioEnabled;
            const btn = document.getElementById('sound-btn');
            const icon = document.getElementById('sound-icon');
            const txt = document.getElementById('sound-text');
            if (audioEnabled) {
                btn.classList.add('sound-on');
                icon.textContent = '🔊';
                txt.textContent = 'صفارة الإنذار مفعّلة';
                playBeep(600, 0.1);
            } else {
                btn.classList.remove('sound-on');
                icon.textContent = '🔇';
                txt.textContent = 'تفعيل صفارة الإنذار التكتيكية';
            }
        }

        function playAlertSiren() {
            if (!audioEnabled || !audioCtx) return;
            const now = audioCtx.currentTime;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(800, now);
            osc.frequency.exponentialRampToValueAtTime(1400, now + 0.3);
            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.4);
            osc.start(now);
            osc.stop(now + 0.4);
        }

        function playBeep(freq, dur) {
            if (!audioCtx) return;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
            osc.start();
            osc.stop(audioCtx.currentTime + dur);
        }

        function switchTab(tab) {
            const videoPanel = document.getElementById('video-panel');
            const imagePanel = document.getElementById('image-panel');
            const tabVideoBtn = document.getElementById('tab-video-btn');
            const tabImageBtn = document.getElementById('tab-image-btn');
            const streamImg = document.getElementById('mjpeg-stream');

            if (tab === 'video') {
                videoPanel.style.display = 'block';
                imagePanel.style.display = 'none';
                tabVideoBtn.classList.add('active');
                tabImageBtn.classList.remove('active');
                streamImg.src = '/video_feed?t=' + new Date().getTime();
            } else {
                videoPanel.style.display = 'none';
                imagePanel.style.display = 'block';
                tabVideoBtn.classList.remove('active');
                tabImageBtn.classList.add('active');
                streamImg.src = '';
            }
        }

        function restartStream() {
            const img = document.getElementById('mjpeg-stream');
            img.src = '/video_feed?t=' + new Date().getTime();
        }

        // Drag & Drop
        const dropzone = document.getElementById('image-dropzone');
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                processFile(files[0]);
            }
        });

        window.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    processFile(items[i].getAsFile());
                    break;
                }
            }
        });

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) processFile(file);
        }

        function processFile(file) {
            showLoading(true);
            const reader = new FileReader();
            reader.onload = async function(e) {
                try {
                    const res = await fetch('/api/detect_image', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: e.target.result })
                    });
                    const data = await res.json();
                    renderAnalysis(data);
                } catch(err) {
                    alert('خطأ أثناء المعالجة: ' + err);
                } finally {
                    showLoading(false);
                }
            };
            reader.readAsDataURL(file);
        }

        async function testSample(name) {
            showLoading(true);
            try {
                const res = await fetch('/api/sample?name=' + name);
                const data = await res.json();
                renderAnalysis(data);
            } catch(e) {
                alert('خطأ: ' + e);
            } finally {
                showLoading(false);
            }
        }

        function renderAnalysis(data) {
            document.getElementById('drop-prompt').style.display = 'none';
            const img = document.getElementById('processed-image');
            img.style.display = 'block';
            img.src = 'data:image/jpeg;base64,' + data.image_base64;

            const drones = data.counts.drones;
            const birds = data.counts.birds;
            document.getElementById('metric-drone-val').textContent = drones;
            document.getElementById('metric-bird-val').textContent = birds;

            // Alert banner & Radar blips
            const alertBanner = document.getElementById('airspace-alert');
            const alertText = document.getElementById('airspace-alert-text');
            const droneBlip = document.getElementById('radar-blip-drone');
            const birdBlip = document.getElementById('radar-blip-bird');

            droneBlip.style.display = drones > 0 ? 'block' : 'none';
            birdBlip.style.display = birds > 0 ? 'block' : 'none';

            if (drones > 0) {
                alertBanner.classList.add('drone-alert');
                alertText.textContent = `تحذير أحمر: تم رصد ${drones} درون معادٍ!`;
                playAlertSiren();
            } else if (birds > 0) {
                alertBanner.classList.remove('drone-alert');
                alertText.textContent = `الأجواء آمنة (تم رصد ${birds} كائنات حية)`;
            } else {
                alertBanner.classList.remove('drone-alert');
                alertText.textContent = 'الأجواء آمنة (لا توجد أهداف)';
            }

            // Optical Forensics List
            const list = document.getElementById('forensics-list');
            const countBadge = document.getElementById('target-count-badge');
            countBadge.textContent = `${data.targets.length} أهداف`;
            list.innerHTML = '';

            if (data.targets.length === 0) {
                list.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 24px; font-size: 13px;">لم يتم رصد أي أهداف في هذا القطاع.</div>';
            } else {
                data.targets.forEach(t => {
                    const isDrone = t.cls === 'drone';
                    const cardClass = isDrone ? 'card-drone' : 'card-bird';
                    const badgeClass = isDrone ? 'badge-drone' : 'badge-bird';
                    const badgeText = isDrone ? '🚨 UAS DRONE' : '🟢 AVIAN BIRD';
                    const morph = t.morph || { vertices: '-', solidity: '-', aspect_ratio: '-' };
                    
                    list.innerHTML += `
                        <div class="forensic-card ${cardClass}">
                            <img class="crop-thumb" src="data:image/jpeg;base64,${t.crop_base64}" alt="Target Crop">
                            <div class="forensic-info">
                                <div class="forensic-header">
                                    <span class="badge-type ${badgeClass}">${badgeText} #${t.id}</span>
                                    <span style="font-family: var(--font-mono); font-weight:700; color: ${isDrone ? '#ff5252' : '#69f0ae'};">
                                        ${(t.conf*100).toFixed(1)}%
                                    </span>
                                </div>
                                <div class="forensic-sub">${t.sub_type}</div>
                                <div class="forensic-meta">
                                    <span>الرؤوس: ${morph.vertices}</span>
                                    <span>الانضغاط: ${morph.solidity}</span>
                                    <span>النسبة: ${morph.aspect_ratio}</span>
                                </div>
                            </div>
                        </div>
                    `;
                });
            }
        }

        function showLoading(show) {
            document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
        }

        window.onload = function() {
            testSample('user_image.jpg');
        };
    </script>
</body>
</html>
"""

def decode_image_bytes(raw_bytes):
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        frame_rgb = np.array(pil_img)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        nparr = np.frombuffer(raw_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def analyze_single_image(img):
    targets = defense_engine.detect_image(img)
    annotated = img.copy()

    targets_data = []
    drone_count = 0
    bird_count = 0

    h, w = img.shape[:2]

    for t in targets:
        x1, y1, x2, y2 = map(int, t.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Extract optical crop with safe margin
        crop = img[max(0, y1-6):min(h, y2+6), max(0, x1-6):min(w, x2+6)]
        if crop.size == 0:
            crop = np.zeros((32, 32, 3), dtype=np.uint8)
        
        _, c_buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        crop_b64 = base64.b64encode(c_buf).decode('utf-8')

        if t.cls == 'drone':
            drone_count += 1
        else:
            bird_count += 1

        targets_data.append({
            'id': f"{t.id:02d}",
            'cls': t.cls,
            'sub_type': t.sub_type,
            'conf': t.conf,
            'box': t.box,
            'morph': t.morph,
            'threat_level': t.threat_level,
            'crop_base64': crop_b64
        })

    hud_img = render_tactical_hud(annotated, targets, show_trails=False)
    _, buf = cv2.imencode('.jpg', hud_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode('utf-8')

    return b64, targets_data, {'drones': drone_count, 'birds': bird_count}

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            if path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode('utf-8'))
            elif path == '/video_feed':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()

                video_path = 'video.mp4'
                if not os.path.exists(video_path):
                    return

                cap = cv2.VideoCapture(video_path)
                defense_engine.reset_tracker()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break

                    targets = defense_engine.process_stream_frame(frame)
                    hud_frame = render_tactical_hud(frame, targets, show_trails=True)
                    _, buffer = cv2.imencode('.jpg', hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(buffer)))
                        self.end_headers()
                        self.wfile.write(buffer.tobytes())
                        self.wfile.write(b"\r\n")
                        time.sleep(0.04)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        break
                cap.release()
            elif path == '/api/sample':
                params = urllib.parse.parse_qs(parsed.query)
                sample_name = params.get('name', ['user_image.jpg'])[0]
                if os.path.exists(sample_name):
                    with open(sample_name, 'rb') as f:
                        raw_b = f.read()
                    img = decode_image_bytes(raw_b)
                    b64, targets, counts = analyze_single_image(img)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'image_base64': b64, 'targets': targets, 'counts': counts}).encode('utf-8'))
                else:
                    self.send_error(404, 'Sample not found')
            elif path == '/output_test.mp4':
                if os.path.exists('output_test.mp4'):
                    with open('output_test.mp4', 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404, 'File not found')
            else:
                self.send_error(404, 'Not Found')
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        if self.path == '/api/detect_image':
            content_length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(content_length)

            img_bytes = None
            try:
                payload = json.loads(raw_body.decode('utf-8'))
                if 'image' in payload:
                    data_str = payload['image']
                    if ',' in data_str:
                        data_str = data_str.split(',', 1)[1]
                    img_bytes = base64.b64decode(data_str)
            except Exception:
                if b'base64,' in raw_body:
                    b64_part = raw_body.split(b'base64,', 1)[1].split(b'"', 1)[0]
                    img_bytes = base64.b64decode(b64_part)
                else:
                    img_bytes = raw_body

            if img_bytes:
                img = decode_image_bytes(img_bytes)
                if img is not None:
                    b64, targets, counts = analyze_single_image(img)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'image_base64': b64, 'targets': targets, 'counts': counts}).encode('utf-8'))
                    return

            self.send_error(400, 'Could not decode image data')
        else:
            self.send_error(404, 'Not Found')

    def log_message(self, format, *args):
        pass

def run_server(port=5000):
    server = ThreadedHTTPServer(('127.0.0.1', port), RequestHandler)
    print(f"==================================================")
    print(f"Military-Grade Air Defense C2 Server is live at:")
    print(f"http://127.0.0.1:{port}")
    print(f"==================================================")
    server.serve_forever()

if __name__ == '__main__':
    run_server(5000)
