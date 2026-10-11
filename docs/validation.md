# Validation and known limits

This is a Windows CPU preview, not a production reliability certification.

Automated tests cover reference validation, profile storage, consent checks, input path restrictions, sentence-aware chunking, pronunciation replacement, history recovery, pitch shifting, streaming exports, mastering signal properties, and worker memory sequencing. They use synthetic fixtures and do not download models.

Real local tests produced voice-cloned audio with both an approximately 29-second reference and a 35-second reference. Matching short narration tests completed in approximately 5.5–6.2 minutes each on an i3-12100 / 16 GB machine, producing about 6.5 seconds of audio. The reference owner preferred the newer sample. These are observations on one machine, not performance guarantees.

A separate real 4,000-word stress test planned 120 segments. It completed seven segments and failed while preparing the reference for segment eight after about 52 minutes, reporting a RuntimeError. Peak observed process-tree working memory was approximately 9.26 GiB and private memory approximately 10.28 GiB. It did not produce a completed 4,000-word file. Earlier attempts also encountered Windows virtual-memory error 1455 and native runtime failures. The exact cause of the stress-test RuntimeError is unresolved.

The UI and short jobs have been tested locally. Full long-form reliability, GPU operation, professional listening quality across scripts, the new GitHub workflow, and installation on a fresh independent Windows machine remain unverified. No personal audio or detailed private diagnostic logs are included in this repository.
