# Unleashed v1.2

A face swap app for images and videos with a browser UI (Gradio): sources from photos or
facesets built from many photos, a choice of which faces to replace, expression restore,
occlusion masks, enhancers, and temporal smoothing for video.

## Running

- **Google Colab (GPU):** open `unleashed_colab.ipynb`, set `BRANCH` if needed and run the
  cells. The UI appears under the *Jalankan Unleashed* cell.
- **Mac (CPU; building facesets, previews):** `./runMacOS.sh`, then open http://127.0.0.1:7860.

Results, *My defaults* and the output options are kept in the output folder (Google Drive on
Colab), so a new session keeps them.

## Responsible use

Use faces only with the consent of the people shown, say that a result is altered when you
share it, and never use this to deceive, harass, or make sexual images of real people.

## License and credits

Unleashed is a modified version of roop-unleashed by C0untFloyd, itself based on roop by
s0md3v, and like them it is licensed under the GNU Affero General Public License v3 (see
`LICENSE`). It has been changed substantially since 2025; the git history lists the changes.

- Mouth restoration: adapted from iVideoGameBoss's DeepFaceCam.
- `unleashed/ffmpeg_writer.py`: MIT license, © 2015 Zulko, © 2023 Janvarev Vladislav.
- The models (face swap, face analysis, expression restore, masks, enhancers) come with their
  own licenses; InsightFace's inswapper, for example, is for non-commercial research only.
