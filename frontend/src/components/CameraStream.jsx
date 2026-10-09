import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { CollapseIcon, EyeIcon, EyeOffIcon, ExpandIcon } from './icons';

// Box colour per person status from the backend pipeline.
const STATUS_COLORS = {
  verified: '#4fd1c5',
  pending: '#f2c14e',
};
const ALERT_COLOR = '#e8613d';

// What the "in view" list says for each status (never colour alone).
function personText(p) {
  if (p.status === 'verified') return p.name;
  if (p.status === 'unauthorized') return `${p.name}: not authorized`;
  return p.label; // "Checking", "Move closer", "Unrecognized", "Liveness failed"...
}

function statusClass(status) {
  if (status === 'verified') return 'verified';
  if (status === 'pending') return 'pending';
  return 'alert';
}

export default function CameraStream({
  zone,
  stats,
  videoHidden,
  onToggleVideo,
  expanded,
  onToggleExpand,
  badgeEnabled,
  onConnectionChange,
  onBadgeDecision,
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(new Image());
  const wsRef = useRef(null);
  const [cameraConnected, setCameraConnected] = useState(false);
  const [hasFrame, setHasFrame] = useState(false);
  const [people, setPeople] = useState([]);
  const peopleKeyRef = useRef('');

  // The socket effect reads these through refs so it only re-runs when the
  // zone changes. Depending on them directly would close and reopen the
  // socket on every parent render or video toggle.
  const onConnectionChangeRef = useRef(onConnectionChange);
  const videoHiddenRef = useRef(videoHidden);
  useEffect(() => {
    onConnectionChangeRef.current = onConnectionChange;
  }, [onConnectionChange]);

  // Tell the backend to stop (or resume) sending frames for this viewer.
  useEffect(() => {
    videoHiddenRef.current = videoHidden;
    api.setStreamVideo(wsRef.current, !videoHidden);
    if (videoHidden) {
      const canvas = canvasRef.current;
      canvas?.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [videoHidden]);

  useEffect(() => {
    let cancelled = false;
    let ws = null;
    let retryTimer = null;

    const drawFrame = (payload) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');

      if (!payload.frame || videoHiddenRef.current) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setHasFrame(false);
        return;
      }

      const img = imageRef.current;
      img.onload = () => {
        if (videoHiddenRef.current) return;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        setHasFrame(true);

        // The canvas is drawn at the stream's resolution and scaled to fit the
        // panel, so size strokes and text in on-screen pixels.
        const unit = canvas.width / Math.max(canvas.clientWidth, 1);
        const fontPx = Math.round(12 * unit);
        for (const det of payload.detections || []) {
          const [x1, y1, x2, y2] = det.bbox;
          const color = STATUS_COLORS[det.status] || ALERT_COLOR;
          const label = det.label || det.status;

          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(2, 2 * unit);
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          ctx.font = `600 ${fontPx}px "IBM Plex Mono", monospace`;
          const textWidth = ctx.measureText(label).width;
          const chipH = fontPx + 8 * unit;
          ctx.fillStyle = color;
          ctx.fillRect(x1, Math.max(0, y1 - chipH), textWidth + 10 * unit, chipH);
          ctx.fillStyle = '#12161b';
          ctx.fillText(label, x1 + 5 * unit, Math.max(fontPx, y1 - 5 * unit));
        }
      };
      img.src = `data:image/jpeg;base64,${payload.frame}`;
    };

    // Only re-render the list when who's in view (or their status) changes,
    // not on every message.
    const updatePeople = (detections) => {
      const key = detections.map((d) => `${d.trackId}:${d.status}:${d.name}:${d.label}`).join('|');
      if (key === peopleKeyRef.current) return;
      peopleKeyRef.current = key;
      setPeople(detections);
    };

    const connect = () => {
      ws = api.openStream(zone, { video: !videoHiddenRef.current });
      wsRef.current = ws;

      ws.onmessage = (event) => {
        if (cancelled) return;
        const payload = JSON.parse(event.data);
        setCameraConnected(payload.connected);
        onConnectionChangeRef.current?.(payload.connected);
        updatePeople(payload.detections || []);
        drawFrame(payload);
      };

      ws.onclose = () => {
        if (cancelled) return;
        // Lost the stream: show this camera as offline rather than its last
        // known state, until the connection comes back.
        setCameraConnected(false);
        onConnectionChangeRef.current?.(false);
        updatePeople([]);
        drawFrame({ frame: null });
        retryTimer = setTimeout(connect, 1500);
      };

      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
      ws?.close();
    };
  }, [zone]);

  const fps = stats?.processing_fps;
  const size = stats?.frame_size;

  return (
    <div className={`panel camera-panel ${expanded ? 'camera-panel--expanded' : ''}`}>
      <div className="panel-header camera-header">
        <div className="camera-title">
          <h2>{zone}</h2>
          <span className={`pill ${cameraConnected ? 'pill--ok' : 'pill--alert'}`}>
            <span className={`status-dot ${cameraConnected ? 'status-dot--ok' : 'status-dot--off'}`} />
            {cameraConnected ? 'Live' : 'Offline'}
          </span>
        </div>
        <div className="camera-tools">
          <button
            className="tool-btn"
            onClick={onToggleVideo}
            aria-pressed={videoHidden}
            title={videoHidden ? 'Show the video again' : 'Stop showing video; recognition keeps running'}
          >
            {videoHidden ? <EyeIcon /> : <EyeOffIcon />}
            {videoHidden ? 'Show video' : 'Hide video'}
          </button>
          <button
            className="tool-btn"
            onClick={onToggleExpand}
            aria-pressed={expanded}
            aria-label={expanded ? `Shrink ${zone} feed` : `Enlarge ${zone} feed`}
            title={expanded ? 'Shrink' : 'Enlarge'}
          >
            {expanded ? <CollapseIcon /> : <ExpandIcon />}
          </button>
        </div>
      </div>

      <div className="camera-frame">
        <canvas ref={canvasRef} className="camera-canvas" />
        {!cameraConnected ? (
          <div className="camera-placeholder">
            <p>Camera not connected</p>
            <p className="camera-placeholder-sub">Waiting for the backend to reach the camera</p>
          </div>
        ) : videoHidden ? (
          <div className="camera-placeholder camera-placeholder--hidden">
            <EyeOffIcon />
            <p>Video hidden</p>
            <p className="camera-placeholder-sub">Recognition and alarms are still running</p>
          </div>
        ) : (
          !hasFrame && (
            <div className="camera-placeholder">
              <p>Starting video…</p>
            </div>
          )
        )}
      </div>

      <div className="camera-footer">
        <ul className="in-view" aria-label={`People in view at ${zone}`}>
          {people.length === 0 ? (
            <li className="in-view-empty">Nobody in view</li>
          ) : (
            people.map((p) => (
              <li key={p.trackId} className="person-chip">
                <span className={`chip-dot chip-dot--${statusClass(p.status)}`} aria-hidden="true" />
                {personText(p)}
              </li>
            ))
          )}
        </ul>
        {fps != null && (
          <span className="camera-stats" title="Frames analysed per second · camera resolution">
            {fps.toFixed(1)} fps{size ? ` · ${size[0]}×${size[1]}` : ''}
          </span>
        )}
      </div>

      {badgeEnabled && <BadgeSimulator zone={zone} onDecision={onBadgeDecision} />}
    </div>
  );
}

// Triggers the same check a real badge reader at this door would, so the
// badge + face flow can be tried before the reader is wired up.
function BadgeSimulator({ zone, onDecision }) {
  const [badgeId, setBadgeId] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    if (!badgeId.trim()) return;
    setBusy(true);
    setResult(null);
    try {
      const r = await api.badge(zone, badgeId.trim());
      setResult(r);
      onDecision?.();
    } catch (err) {
      setResult({ decision: 'error', reason: err.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="badge-sim" onSubmit={submit}>
      <input
        value={badgeId}
        onChange={(e) => setBadgeId(e.target.value)}
        placeholder="Badge ID"
        aria-label={`Badge ID to test at ${zone}`}
      />
      <button type="submit" className="btn btn-ghost btn-small" disabled={busy || !badgeId.trim()}>
        {busy ? 'Checking…' : 'Test badge'}
      </button>
      {result && (
        <span className={`badge-result decision--${result.decision}`} role="status">
          <span aria-hidden="true">{result.decision === 'grant' ? '✓' : '✕'}</span>
          {result.decision === 'grant' ? `Granted: ${result.name}` : `Denied: ${result.reason}`}
          {result.matchScore != null && ` (match ${result.matchScore.toFixed(2)})`}
        </span>
      )}
    </form>
  );
}
