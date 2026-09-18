# Product

<!-- nikhil:product-schema 1 -->

## Platform

web

## Users

One person, the owner, on their own Windows laptop. They are going through personal photos and screenshots before posting or sending them, and want anything private blacked out first. They are not a privacy researcher and should never need to learn detector or policy vocabulary to finish the job.

## Product Purpose

Photo in, private details covered in solid black, clean file out. ConsentGuard finds faces, number plates, text, handwriting, QR codes and hidden file data, covers them, lets the owner fix what it missed or over-covered, then writes a brand-new file with location and camera data removed and re-checks that file before offering the download.

Success: the owner gets a correctly covered photo in one or two clicks, and always looks at the result before sharing it.

## Positioning

Everything runs locally on 127.0.0.1. The photo never leaves the machine, the working copy is deleted when the tab closes, and the saved file is re-encoded from scratch and re-scanned by the same detectors before download. Detection is automatic but openly incomplete; human review is the safety net.

## Operating Context

- Served by `main_project/scripts/stage_05_review_export/run_web_app.py`, opened in Chrome or Edge on the laptop.
- Two policy modes share one interface. **Personal** (the shipped default) unlocks download after the owner reviews. **Research** keeps consent forms and release gates; it is parked, but must keep working.
- One photo per session: upload, detect, check the cover, save, download.

## Capabilities and Constraints

- Accepts JPEG, PNG and WebP stills up to the configured size (25 MB by default).
- One-click "Erase and save", or "Check it myself first" to edit the cover with brush and eraser.
- Some detection types are known to be weak (for example text and handwriting); the interface flags them "check manually" instead of blocking the download.
- The saved file is checked for: opening correctly, being a fresh encode, no location/camera data, and a re-scan for faces, plates, text and barcodes. All checks stay visible on the result screen.
- Do not reintroduce release-gate blocking, consent-record forms or research jargon into personal mode.

## Brand Commitments

- Name: ConsentGuard.
- Voice: plain, direct, honest about limits. "It will miss things" is a feature, not a disclaimer to hide.

## Evidence on Hand

- No testimonials, customers or accuracy claims exist. Do not invent detection rates or guarantees.
- Measured coverage lives in the research reports; the interface does not quote numbers.

## Product Principles

1. Plain words over system words. Every status, check and warning reads as a sentence the owner would say.
2. The photo is the subject. Chrome stays out of its way; the cover shown while editing matches what gets saved.
3. Honest limits, placed where the decision happens. The reminder to look before sharing sits next to the download, not on every screen.
4. Local and disposable. Nothing persists beyond the tab.
