# Troubleshooting

## First launch

Download the Qwen voice model in Settings before generating. Whisper Base is optional; without it, enter the reference transcript yourself. Model setup needs internet and around 25 GB of free working space. A GitHub source archive is not a Windows installer.

## Slow or failed generation

CPU inference can take minutes for seconds of speech. Long inputs are split into segments, but each segment still needs enough memory to load and run the model. Close memory-heavy applications and avoid concurrent generation jobs.

Windows error 1455 means insufficient virtual memory. Consider a system-managed paging file under Advanced system settings → Performance → Advanced → Virtual memory, and restart if Windows requests it. The app does not change these settings itself.

A generic RuntimeError or native exit code does not by itself establish an out-of-memory cause. The latest local voice-worker.log can help diagnose the stage. Review it for local paths before sharing; never attach voice recordings or transcripts without intending to disclose them. The reported 4,000-word stress test did not complete; see validation.md.

## Voice does not sound right

Use one speaker in a quiet room, without music or effects. Prefer an original WAV recording. Correct every word of the reference transcript and use the delivery you want the model to imitate. Deep Voice is a pitch effect and can reduce perceived naturalness; test with it off first. Pronunciation replacements change synthesis text, not the reference transcript. Filters cannot repair every model pronunciation or delivery error.

## History, exports and cancellation

Completed WAV files are saved locally and listed in History. Existing older files can be recovered into the list without their original text. Files over 40 MiB can be exported for external playback. Exports require a new filename and never overwrite an existing file.

Cancel stops the worker process tree. Failed jobs are not listed as completed generations. Abrupt termination can leave a private temporary generation folder. Do not remove folders belonging to an active job.

## Reporting a bug

Include the app version, Windows version, RAM, compute device, approximate text length, failed stage and exact error. Reproduce using nonprivate example text. Leave personal audio, reference transcripts and data-folder contents out of public issues.
