# Gatekeeper dashboard

React frontend for the access-control system. Talks to `server.py` (the
FastAPI backend one directory up) over REST and a WebSocket — it doesn't
run any face recognition itself.

## Setup

```bash
npm install
npm run dev
```

Opens at `http://localhost:5173`. The backend must be running separately
at `http://localhost:8000` (see the top-level README) — CORS is already
configured on the backend for this dev server's origin.

On first load the dashboard asks for the backend's API token (the value of
`GATEKEEPER_API_TOKEN`, or the one the backend printed at startup). It's kept
in `localStorage`, and if the backend later rejects it, for example after a
restart with a new generated token, the dashboard returns to the sign-in prompt.

If you deploy the backend somewhere other than `localhost:8000`, update
`API_BASE` in `src/api.js`.

## How it's structured

| File | Purpose |
|---|---|
| `src/api.js` | All backend calls in one place — REST requests, the stream socket (including the video on/off message), and authenticated image loading. Also holds the API token |
| `src/App.jsx` | Sign-in gate, then the header summary, polling (status every 5 s doubles as the server heartbeat, alarms every 4 s), warning banners, which feeds have video hidden, and modal state |
| `src/components/TokenPrompt.jsx` | Asks for the API token and checks it against the backend |
| `src/components/CameraStream.jsx` | One camera: draws each frame with each person's status, the in-view list, frame rate, the hide-video and enlarge buttons, and the "Test badge" box when badge checks are on |
| `src/components/StaffPanel.jsx` | Searchable roster: zones, photos, capture at a door camera, suspend/reinstate/remove (badge IDs when badge checks are on) |
| `src/components/EnrollModal.jsx` | Enroll a new staff member from name, zones and photos |
| `src/components/PhotoManagerModal.jsx` | View, add, and delete an existing staff member's enrolled photos |
| `src/components/AlarmLog.jsx` | Alarms with severity, snapshot, open/all filter and acknowledgement |
| `src/components/AccessLog.jsx` | Badge decisions with the frame each was based on (only when badge checks are on) |
| `src/components/AuthedImage.jsx` | Shows an image that's behind the API token (photos, snapshots) |
| `src/components/icons.jsx` | Small stroke icons used on buttons |
| `src/format.js`, `src/prefs.js` | Time formatting, and per-browser preferences (stored safely in `localStorage`) |

## Notes on the live feed

Frames arrive over the WebSocket as base64 JPEG plus a `detections` array.
Each entry has the bounding box, the person's `status` (`pending`,
`verified`, `unauthorized`, `unknown`, `spoof_suspected`, `unverifiable`)
and a ready-made `label`. The box and label drawing happens entirely
client-side in `CameraStream.jsx`, not baked into the video, so boxes stay
crisp at any display size. Strokes and text are sized in on-screen pixels,
whatever the stream's resolution.

Browsers can't attach headers to a WebSocket, so the client sends the API
token as the first message after the socket opens. The backend closes the
socket with code 1008 if the token is missing or wrong.

After the token the client can send `{"video": false}` (or `true`) at any
time. With video off the backend stops sending frames, and stops encoding
them if no other dashboard is watching that camera. It keeps sending
`connected` and `detections` a few times a second, which is how the in-view
list keeps working while the picture is hidden. Recognition and alarms run on
the server regardless.

The camera connection state comes from the same WebSocket message
(`payload.connected`), so the "Offline" state reflects the backend's actual
camera state. If the socket itself drops, the feed also shows offline until it
reconnects.
