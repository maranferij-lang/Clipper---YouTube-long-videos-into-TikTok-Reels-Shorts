Selection standard

The rules the scorer and the human reviewer apply when choosing moments from a long video.

Rubric, 0 to 100
Hook, Flow, Value and Trend are scored 0 to 25 each. Penalties apply for jargon, out of context fragments, digressions, unexplained references and cross talk. A moment is a candidate at 65 or more and only when it contains one complete arc: setup, development, payoff. Do not pad a batch with weak clips to reach a number. Zero clips from a video is a valid answer.

Boundaries
1. Start on the sentence that names the subject, never in the middle of a thought and never on a fragment of the previous phrase.
2. End on a finished sentence. A closing line from the host can be the payoff.
3. Floor of about 40 seconds, no hard ceiling. Cut water inside with several ranges, splices go forward only.
4. Do not relocate a quote or a number to the front as a cold open. The viewer needs context first. The hook is the first sentence of the clip itself.
5. The first seconds must show the speaker's face, not b-roll or a screenshot.
6. Move range edges into silence, auto-caption timestamps usually lag the real word start by roughly 0.08 s.

Picture
1. Source quality matters: use the highest resolution available, never a 360p source.
2. One speaker fills the frame. The camera follows the active speaker, switches only when the other person speaks in whole sentences, and ignores short interjections.
3. Wide shots are shown whole on a blurred background, or not used. Never crop into a wide shot.
4. Hold any angle for at least 2 seconds, avoid flipping the frame for a second and coming back.
5. Verbatim subtitles, two or three words per line, bold font, lower third.
6. Audio: high-pass at 85 Hz, loudness normalised to -15 LUFS with a true peak of -1.5 dB.

Quality control
qccheck_v2.py runs on every clip before delivery: hold times, scale jumps, band share, cropped boundaries, resolution, loudness, subtitle length and small face crops. Defects are caught by the check, not by the viewer.
