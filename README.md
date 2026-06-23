# Reaction Dot Game

A React + Vite + TypeScript web app for a reaction-time dot game with browser camera permission, local MediaPipe face/eye landmark tracking, consent flow, trial data capture, and downloadable results.

## What it does

- Shows red, green, blue, and yellow dots.
- Randomises dot size, position, and display duration.
- Measures click/tap reaction time.
- Requests camera permission after consent.
- Uses MediaPipe Face Landmarker locally in the browser.
- Records approximate iris, gaze proxy, and head-position landmark data.
- Shows a results breakdown by colour and size.
- Exports all data as JSON, trials CSV, and eye-samples CSV.

## Important note

This is not clinical or lab-grade eye tracking. Webcam-based face/eye landmark data is approximate and device-dependent.

The app currently stores data in browser memory and lets the participant download it at the end. It does not upload results to a backend. If you want central data collection, add a server endpoint or connect a service such as Firebase or Supabase.

## Requirements

- Node.js 20+ recommended
- npm
- A modern browser
- HTTPS for camera access when deployed publicly. Localhost is allowed for development.

## Run locally

```bash
npm install
npm run dev
```

Then open the local URL shown in the terminal, usually:

```bash
http://localhost:5173
```

## Build for deployment

```bash
npm run build
npm run preview
```

The production build will be in:

```bash
dist
```

## Deploy

You can deploy this to Vercel, Netlify, Firebase Hosting, or another HTTPS host.

For Vercel:

```bash
npm install -g vercel
vercel
```

For Netlify:

```bash
npm install -g netlify-cli
netlify deploy
```

## Data files

At the end of the game, participants can download:

- `reaction-dot-game-<participant>.json`: full experiment payload
- `reaction-trials-<participant>.csv`: trial-level reaction data
- `reaction-eye-samples-<participant>.csv`: frame-level eye/face landmark samples

## Privacy reminder

Before using this with real participants, add your own participant information sheet, consent wording, ethics/privacy review, and data-retention policy. Do not collect identifiable information unless you genuinely need it.
