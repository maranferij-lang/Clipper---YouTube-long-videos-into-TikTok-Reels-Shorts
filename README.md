Clipper

Turns long videos and podcasts into vertical 9:16 clips for TikTok, Reels and Shorts. It scores candidate moments, cuts on sentence boundaries, follows the active speaker, adds subtitles and checks quality before delivery. Runs on your own hardware, no watermark and no limits.

Pipeline
1. opus_scorer_v22.py reads word level timings and ranks candidate moments with a transparent rubric (see docs/selection-standard.md). It is a first pass, the final choice and boundaries are a human decision.
2. pipeline_v3.py downloads only the needed ranges in high resolution, cuts them, runs active speaker detection (Light-ASD on CPU), plans the camera path, renders, normalises audio and runs QC. It is resumable: state is kept in state.json, call it again until it prints ALL DONE.
3. render_opus_tc_v4.py is the renderer: speaker crop, blurred band layout for wide shots, karaoke style subtitles.
4. qccheck_v2.py catches defects before the clip leaves the machine.

Setup on macOS
1. bash tools/setup_mac_native.sh
2. bash tools/get_models.sh
3. Put a bold font at tools/Mv.ttf
4. Write a jobs.json (docs/example_jobs.json) with the YouTube id, the word timings file and the ranges in absolute seconds.
5. bash tools/run_loop.sh path/to/jobs.json

Third party pieces you install separately: ffmpeg, yt-dlp, PyTorch, OpenCV, Light-ASD (Junhua-Liao/Light-ASD) and the YuNet model from OpenCV Zoo. Check their own licences.

Notes
Code comments are in English. The scoring lexicons in opus_scorer_v22.py target Ukrainian speech (plus some Russian). Only use it on video you have the right to repost.

License: MIT
