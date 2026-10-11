# Put OpenVoice4U on GitHub

## Upload the clean source

Use the `OpenVoice4U-source` folder created by `scripts/export_source.py`, not the personal development folder or a used portable app directory. The export includes source, tests, licenses and workflows; it excludes recordings, transcripts, model weights, Python environments, installed apps and machine-specific acceptance scripts.

1. Create a new public GitHub repository named `openvoice4u`. Leave the automatic README/license options unchecked because these files are already provided.
2. Extract `OpenVoice4U-source.zip` locally. Upload the **contents** of `OpenVoice4U-source` to the repository root. Uploading the zip file itself does not create a usable source repository.
3. Ensure `.github/workflows` and the dotfiles are included. Using Git or GitHub Desktop avoids file-picker omissions.

For Git, run these commands inside the extracted source folder. Replace YOUR-USERNAME first:

```powershell
git init -b main
git add .
git commit -m "Initial open-source release"
git remote add origin https://github.com/YOUR-USERNAME/openvoice4u.git
git push -u origin main
```

## Build a downloadable installer

Read [the publication review](release-review.md) first. The current preview has unresolved dependency advisories and pending binary redistribution checks. Building an artifact is not approval to publish it as a stable installer.

Open **Actions → Build Windows installer → Run workflow** on the default branch. Wait for success, then download the **OpenVoice4U-Windows-x64** artifact. It contains the setup executable, checksum file, source archive, and dependency inventory. This workflow does not download voice models or access your computer's recordings.

Run the installer on a clean Windows machine, install models through Settings, and test uploading a new reference, transcript correction, generation, History, export, and cancellation. Review third-party binary notices and source obligations for the actual dependencies. The first GitHub-hosted build and clean-machine install are still pending until you run them.

## Create a release

The version must agree in package.json, package-lock.json, pyproject.toml, src-tauri/Cargo.toml, Cargo.lock and tauri.conf.json. For this source package:

```powershell
git tag v0.1.2
git push origin v0.1.2
```

The tag workflow builds the installer and creates a **draft prerelease** with its assets. Review it in Releases and publish it when ready. Users can then download the setup.exe directly without Git or developer tools. Never upload your personal `OpenVoice4U` directory, model cache, `.local` folder, or voice samples as release assets.

The workflow uses GitHub's repository token, not a personal token embedded in source. Repository policies can require Actions approval or restrict release creation. Enable private vulnerability reporting under Settings → Code security if available.

## Release checklist

- Automated checks pass; installer and voice workflow tested on clean Windows.
- Version and tag agree; SHA-256 checksums supplied.
- Dependency/native-library notices and redistribution obligations reviewed.
- No user audio, profiles, settings, transcripts, secrets or models in source/assets.
- Known CPU speed, memory, long-form reliability and preview limits disclosed.
- Explain whether binaries are signed. Signing credentials belong in repository secrets, never source.

The workflow follows [Tauri's Windows packaging](https://v2.tauri.app/distribute/windows-installer/) and [GitHub's manual workflow mechanism](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
