# Dataset access requests

Drafts for datasets that need a signed agreement. **Nothing here has been sent.**
You send these yourself, from your own university account.

## Read this before sending anything

**1. Both datasets are for academic research only.** Each agreement has you
state that the data will be used only for non-commercial research, never
redistributed, and never sold or profited from. Send them only if that is true
of how you will use ConsentGuard. If the project is purely a personal tool with
no research purpose, signing either agreement would misstate your use — in that
case, skip them and annotate your own road photos instead.

**2. They must come from your university address.** The authors reject requests
from personal addresses such as Gmail. Use the `.edu` / `.ac.in` account.

**3. They help less than the sizes suggest.** Both are Brazilian/Mercosur plates.
They teach plate *shape and position* at more distances and angles, not Indian
plate appearance. RodoSol comes from fixed toll-booth cameras, so its plates
are mostly close and clear; UFPR-ALPR, filmed from a moving car, is the one that
actually covers the distant-plate failure. If you only send one, send UFPR-ALPR.

Obligations that come with the data:

- No redistribution and no modification of the dataset itself. It stays in
  `data/raw/`, which Git ignores.
- Images may appear only in academic publications and presentations — not in
  the ConsentGuard README, the status page, or a demo video.
- Any model trained on it inherits the non-commercial restriction.
- Cite the paper named in each agreement if you publish anything using it.
- Record it in `THIRD_PARTY.md` when the data arrives.

## Files

| File | Dataset | What you do |
|---|---|---|
| [`ufpr_alpr_request_email.md`](ufpr_alpr_request_email.md) | UFPR-ALPR — 4,500 images, 1920×1080, moving camera in real traffic | Fill in the five fields, send the email |
| [`rodosol_alpr_request_email.md`](rodosol_alpr_request_email.md) | RodoSol-ALPR — 20,000 images, 1280×720, toll-booth cameras | Print the PDF, sign it **by hand**, scan it, attach it |

Both go to the first author, Rayson Laroca, at `rblsantos@inf.ufpr.br`. The
usual turnaround is 1–5 business days; check your spam folder if nothing
arrives.

When a download link comes back, forward it to me and I will pull the data
into `data/raw/`, convert it with the existing specialist pipeline, and measure
plate coverage before and after.
