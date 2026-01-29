# Git LFS Setup for Streamlit App

## Install Git LFS

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Or download from: https://git-lfs.com/
```

## Initialize Git LFS

```bash
cd /home/melvin/projects/flora/.worktrees/streamlit-app
git lfs install
```

## Track thumbnail files

```bash
git lfs track "streamlit_app/thumbnails/*.jpg"
git add .gitattributes
```

## Verify LFS tracking

```bash
git lfs track  # Should show: streamlit_app/thumbnails/*.jpg
```

## Add and commit

```bash
git add streamlit_app/thumbnails/
git commit -m "Add thumbnails with LFS"
```

## Notes

- Thumbnails total size: ~23 MB
- Git LFS makes these files efficient to clone/pull
- On GitHub, LFS provides 1GB free storage + 1GB free bandwidth/month
- Streamlit Cloud supports Git LFS out of the box
