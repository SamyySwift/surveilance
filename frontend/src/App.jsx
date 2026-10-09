import { useCallback, useEffect, useState } from 'react';
import { api, getToken, onUnauthorized, setToken } from './api';
import CameraStream from './components/CameraStream';
import StaffPanel from './components/StaffPanel';
import AlarmLog from './components/AlarmLog';
import AccessLog from './components/AccessLog';
import EnrollModal from './components/EnrollModal';
import PhotoManagerModal from './components/PhotoManagerModal';
import TokenPrompt from './components/TokenPrompt';
import { EyeIcon, EyeOffIcon } from './components/icons';
import { loadPref, savePref } from './prefs';
import { formatTime } from './format';

export default function App() {
  const [authed, setAuthed] = useState(() => Boolean(getToken()));
  const [notice, setNotice] = useState(null);

  // While signed in, any 401 or rejected stream drops back to the sign-in
  // prompt, which unmounts the dashboard and with it every poll and camera
  // socket. (The prompt reports its own rejections, so no handler then.)
  useEffect(() => {
    if (!authed) return undefined;
    return onUnauthorized(() => {
      setToken(null);
      setAuthed(false);
      setNotice('Your saved token was rejected. The backend may have restarted with a new one.');
    });
  }, [authed]);

  if (!authed) {
    return (
      <TokenPrompt
        notice={notice}
        onAuthenticated={() => {
          setNotice(null);
          setAuthed(true);
        }}
      />
    );
  }

  return <Dashboard />;
}

const HIDDEN_VIDEO_KEY = 'gatekeeper.hiddenVideo';

function Dashboard() {
  const [staff, setStaff] = useState([]);
  const [alarms, setAlarms] = useState([]);
  const [accessEvents, setAccessEvents] = useState([]);
  const [status, setStatus] = useState(null);
  const [serverDownSince, setServerDownSince] = useState(null);
  const [zones, setZones] = useState([]);
  const [cameraConnections, setCameraConnections] = useState({});
  const [hiddenVideo, setHiddenVideo] = useState(() => loadPref(HIDDEN_VIDEO_KEY, []));
  const [expandedZone, setExpandedZone] = useState(null);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [managePhotosFor, setManagePhotosFor] = useState(null);

  // Badge features stay hidden until the backend confirms they're switched on.
  const badgeEnabled = status?.badgeChecks === true;

  const refreshStaff = useCallback(async () => {
    try {
      setStaff(await api.listStaff());
    } catch (err) {
      console.error('Failed to load staff:', err);
    }
  }, []);

  const loadZones = useCallback(async () => {
    try {
      setZones(await api.listZones());
    } catch (err) {
      console.error('Failed to load zones:', err);
    }
  }, []);

  useEffect(() => {
    refreshStaff();
    loadZones();
  }, [refreshStaff, loadZones]);

  const refreshActivity = useCallback(async () => {
    try {
      const [alarmList, events] = await Promise.all([
        api.listAlarms(),
        badgeEnabled ? api.listAccessEvents() : Promise.resolve([]),
      ]);
      setAlarms(alarmList);
      setAccessEvents(events);
    } catch (err) {
      console.error('Failed to load alarms:', err);
    }
  }, [badgeEnabled]);

  useEffect(() => {
    const first = setTimeout(refreshActivity, 0);
    const id = setInterval(refreshActivity, 4000);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, [refreshActivity]);

  // Also the heartbeat: if this fails the dashboard says so, rather than
  // quietly showing stale alarms and frozen video.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await api.status();
        if (!cancelled) {
          setStatus(s);
          setServerDownSince(null);
        }
      } catch (err) {
        console.error('Failed to load status:', err);
        if (!cancelled) setServerDownSince((since) => since ?? Date.now() / 1000);
      }
    };
    load();
    const id = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const openAlarms = alarms.filter((a) => !a.acknowledgedAt).length;

  // Show the open alarm count in the browser tab, so it's visible from other tabs.
  useEffect(() => {
    document.title = openAlarms > 0 ? `(${openAlarms}) Gatekeeper` : 'Gatekeeper — Access Control';
  }, [openAlarms]);

  // Errors reach the dialog that asked, which shows them and stays open.
  const acknowledge = async (alarmId, note) => {
    try {
      await api.ackAlarm(alarmId, note);
    } finally {
      refreshActivity();
    }
  };

  const clearAlarms = async (upToId) => {
    try {
      await api.clearAlarms(upToId);
    } finally {
      refreshActivity();
    }
  };

  const setHidden = (next) => {
    setHiddenVideo(next);
    savePref(HIDDEN_VIDEO_KEY, next);
  };
  const toggleVideo = (zone) =>
    setHidden(hiddenVideo.includes(zone) ? hiddenVideo.filter((z) => z !== zone) : [...hiddenVideo, zone]);
  const allHidden = zones.length > 0 && zones.every((z) => hiddenVideo.includes(z));
  const toggleAllVideo = () => setHidden(allHidden ? [] : [...zones]);

  const online = zones.filter((z) => cameraConnections[z]).length;
  const banner = livenessBanner(status, badgeEnabled);
  const emailProblem = serverDownSince ? null : status?.email?.problem;
  const liveness = serverDownSince ? null : livenessChip(status);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <h1>Gatekeeper</h1>
        </div>

        <div className="header-status">
          <span className={`chip ${online < zones.length ? 'chip--alert' : ''}`} title="Cameras sending video">
            <span className="chip-label">Cameras</span>
            {zones.map((zone) => (
              <span key={zone} className="chip-camera" title={`${zone}: ${cameraConnections[zone] ? 'online' : 'offline'}`}>
                <span className={`status-dot ${cameraConnections[zone] ? 'status-dot--ok' : 'status-dot--off'}`} />
                {zone}
              </span>
            ))}
          </span>
          {liveness && (
            <span className={`chip chip--${liveness.level}`} title="Photo and screen detection">
              <span className="chip-label">Liveness</span>
              {liveness.text}
            </span>
          )}
          <span className={`chip ${openAlarms > 0 ? 'chip--alert' : ''}`}>
            <span className="chip-label">Alarms</span>
            {openAlarms} open
          </span>
          {zones.length > 0 && (
            <button className="tool-btn" onClick={toggleAllVideo} aria-pressed={allHidden}>
              {allHidden ? <EyeIcon /> : <EyeOffIcon />}
              {allHidden ? 'Show all video' : 'Hide all video'}
            </button>
          )}
        </div>
      </header>

      {serverDownSince && (
        <div className="system-banner system-banner--critical" role="alert">
          <span aria-hidden="true">⚠</span> Can't reach the Gatekeeper server since {formatTime(serverDownSince)}.
          Alarms and video on this screen are not updating. Retrying every few seconds.
        </div>
      )}

      {banner && (
        <div className={`system-banner system-banner--${banner.level}`} role="alert">
          <span aria-hidden="true">⚠</span> {banner.text}
        </div>
      )}

      {emailProblem && (
        <div className="system-banner system-banner--warning" role="alert">
          <span aria-hidden="true">⚠</span> Alarm emails aren't being delivered: {emailProblem} Alarms are still
          recorded and shown here.
        </div>
      )}

      <main className="layout">
        <section className="col-primary">
          <div className="camera-grid">
            {zones.map((zone) => (
              <CameraStream
                key={zone}
                zone={zone}
                stats={serverDownSince ? undefined : status?.zones?.[zone]}
                videoHidden={hiddenVideo.includes(zone)}
                onToggleVideo={() => toggleVideo(zone)}
                expanded={expandedZone === zone}
                onToggleExpand={() => setExpandedZone(expandedZone === zone ? null : zone)}
                badgeEnabled={badgeEnabled}
                onConnectionChange={(connected) =>
                  // Return the same object when nothing changed, so a frame
                  // arriving doesn't re-render the whole dashboard.
                  setCameraConnections((prev) =>
                    prev[zone] === connected ? prev : { ...prev, [zone]: connected }
                  )
                }
                onBadgeDecision={refreshActivity}
              />
            ))}
          </div>
          <AlarmLog
            alarms={alarms}
            onAcknowledge={acknowledge}
            onClear={clearAlarms}
            emailSettings={status?.email}
          />
          {badgeEnabled && <AccessLog events={accessEvents} />}
        </section>

        <section className="col-secondary">
          <StaffPanel
            staff={staff}
            zones={zones}
            badgeEnabled={badgeEnabled}
            onRefresh={refreshStaff}
            onEnroll={() => setEnrollOpen(true)}
            onManagePhotos={(member) => setManagePhotosFor(member)}
          />
        </section>
      </main>

      {enrollOpen && (
        <EnrollModal
          availableZones={zones}
          badgeEnabled={badgeEnabled}
          onClose={() => setEnrollOpen(false)}
          onEnrolled={() => {
            setEnrollOpen(false);
            refreshStaff();
          }}
        />
      )}

      {managePhotosFor && (
        <PhotoManagerModal
          staff={managePhotosFor}
          onClose={() => setManagePhotosFor(null)}
          onChanged={refreshStaff}
        />
      )}
    </div>
  );
}

// Short liveness state for the header.
function livenessChip(status) {
  const live = status?.liveness;
  if (!live) return null;
  if (live.mode === 'enforce') return live.available ? { level: 'ok', text: 'On' } : { level: 'alert', text: 'Unavailable' };
  if (live.mode === 'monitor') return { level: 'warning', text: 'Monitor only' };
  return { level: 'warning', text: 'Off' };
}

// A standing warning when liveness isn't protecting the doors.
function livenessBanner(status, badgeEnabled) {
  const live = status?.liveness;
  if (!live) return null;
  if (live.mode === 'enforce' && !live.available) {
    const effect = badgeEnabled
      ? 'so every badge entry is being denied'
      : 'so photos and screens held up to the cameras are not being detected';
    return {
      level: 'critical',
      text: `Liveness check unavailable, ${effect}.${live.problem ? ` ${live.problem}` : ''}`,
    };
  }
  if (live.mode === 'monitor') {
    return { level: 'warning', text: 'Liveness is in monitor mode: photos and screens are recorded but not blocked.' };
  }
  if (live.mode === 'off') {
    return { level: 'warning', text: 'Liveness is off: photos and screens held up to the camera are not checked.' };
  }
  return null;
}
