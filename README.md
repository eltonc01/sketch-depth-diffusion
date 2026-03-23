# Sketch-Depth Diffusion Website (gh-pages)

This branch hosts the static GitHub Pages website for Sketch-Depth Diffusion.

## Contents
- `index.html`: project page
- `static/`: CSS/JS/images/videos from the Nerfies-style template
- `.gitignore`: branch-level ignores

## Local preview
Run from this folder:

```bash
python -m http.server 8000
```

Open:
- http://localhost:8000

## Update workflow
1. Replace placeholder text in `index.html`.
2. Replace media in `static/images` and `static/videos`.
3. Commit and push to `gh-pages`.

## GitHub Pages settings
In the repo settings:
- Source: Deploy from a branch
- Branch: `gh-pages`
- Folder: `/` (root)

# Website Template
Based on [Nerfies](https://github.com/nerfies/nerfies.github.io)!
