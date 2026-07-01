# Authoring Guide — Blog Posts, Demos & Publishing

Everything you need to write a new blog post, preview it, and get it live on
<https://eitamar-saraf.github.io>. This is a hand-built [Astro](https://astro.build)
site (static output) styled with Tailwind CSS v4. **Blog posts are plain `.astro`
files — not Markdown/MDX** — so you get full HTML + Tailwind control in each post.

---

## 0. The two-repo workflow (read this first)

There is **no dedicated "blog" repo**. A published post spans **two** repositories:

1. **This repo — the website (`eitamar-saraf/eitamar-saraf.github.io`).**
   It's the *whole portfolio*: home, CV, projects, demos, **and** the blog. The
   blog is just a section (`src/pages/blog/`). Each post's **writeup** lives here,
   together with a **copy** of the figures it displays (`public/plots/<slug>/`).

2. **One repo per experiment** — e.g. `eitamar-saraf/BN-vs-NonBN` (the BatchNorm
   study), `eitamar-saraf/act-bench` (activation-functions WIP). These hold the
   **research code** that trains the models and generates the plots/dashboards.

The experiment repo *produces* the assets; the website repo *displays* them and
holds the prose. So the figures are generated in the experiment repo and **copied
across** into the website repo — they are not shared or symlinked.

```
experiment repo (e.g. BN-vs-NonBN)            website repo (eitamar-saraf.github.io)
  train.py / callbacks.py ──run──▶ plots ──copy──▶ public/plots/<slug>/*.png
  make_dashboard.py       ──run──▶ dashboard.html ─copy─▶ public/<slug>/...
                                                   src/pages/blog/<slug>.astro   ← the writeup
                                                   src/pages/demos/<slug>.astro  ← optional demo iframe
                                                   └─ a "Repository" button links back to the experiment repo
```

**Two repos ⇒ two separate pushes.** Commit and push the experiment repo on its
own (its code/plots), and commit and push the website repo on its own (the post +
copied assets). The auth setup in §1.3 applies to **both** — each has its own
remote, and the same `gh`/HTTPS fix is needed for each (run the `git remote
set-url` step inside each repo). Only the **website** repo auto-deploys to the
live site (§5).

Concretely, to ship the BatchNorm post:
```bash
# 1. In the experiment repo: generate and commit the artifacts
cd ~/code/BN-vs-NonBN
python train.py --sweep --epochs 20        # produces runs/…
python make_plots.py                       # produces the .png figures
python make_dashboard.py --root runs_bn_probe/ --out bn_dashboard.html
git add -A && git commit -m "…" && git push origin main

# 2. Copy the artifacts into the website repo
cp path/to/figures/*.png ~/code/eitamar-saraf.github.io/public/plots/bn/
cp bn_dashboard.html      ~/code/eitamar-saraf.github.io/public/bn/

# 3. In the website repo: write the post and publish (see §4–§5)
cd ~/code/eitamar-saraf.github.io
# …create src/pages/blog/<slug>.astro, add the index card, then:
git add -A && git commit -m "blog(<slug>): …" && git push origin main
```

The rest of this guide is about **step 3 — the website repo.** Authoring the
experiment itself lives in that experiment's own README.

---

## 1. One-time setup

### 1.1 Node version (important)
Astro 5 requires **Node ≥ 18.20.8**. Check yours:

```bash
node -v
```

If it's older (e.g. 18.19.x), the dev server and build will refuse to run.
Install a modern Node with [nvm](https://github.com/nvm-sh/nvm):

```bash
nvm install 20
nvm use 20
```

### 1.2 Install dependencies

```bash
npm install        # or: npm ci   (clean install from package-lock.json)
```

### 1.3 GitHub account & auth (read this — there's a gotcha)

The repos are owned by the GitHub account **`eitamar-saraf`** (with a hyphen).
On at least one machine the SSH key authenticates as a *different* account
(`eitamarSaraf`, no hyphen) that has **no write access**, so an SSH `git push`
fails with:

```
ERROR: Permission to eitamar-saraf/eitamar-saraf.github.io.git denied to eitamarSaraf.
```

**Fix: push over HTTPS using the GitHub CLI (`gh`) as the `eitamar-saraf` account.**

```bash
# 1. Log in (once). Choose HTTPS when asked for the git protocol.
gh auth login

# 2. Make sure the ACTIVE account is the repo owner:
gh auth status                       # see which account is "Active account: true"
gh auth switch --user eitamar-saraf  # if it isn't

# 3. Let git use gh as its credential helper for github.com:
gh auth setup-git

# 4. Point the remote at the HTTPS URL (not git@... SSH):
git remote set-url origin https://github.com/eitamar-saraf/eitamar-saraf.github.io.git
```

After this, `git push` works and uses the `eitamar-saraf` token automatically —
no password prompts.

> If you have a working SSH key that *is* tied to `eitamar-saraf`, you can keep
> the `git@github.com:...` remote instead and skip the HTTPS steps.

---

## 2. Local preview

```bash
npm run dev        # http://localhost:4321  — live reload while you edit
npm run build      # production build into ./dist
npm run preview     # serve the built ./dist locally
```

Always preview with `npm run dev` before pushing. The production deploy runs the
same `npm run build`, so if `build` is clean locally it will deploy cleanly.

---

## 3. Where things live

```
src/
  pages/
    blog.astro            # the /blog index — a MANUAL grid of post cards
    blog_posts.astro      # /blog_posts — the "ideas" backlog list
    blog/
      batchnorm.astro     # a post  → served at /blog/batchnorm
      xor.astro           # a post  → served at /blog/xor
    demos/
      batchnorm.astro     # optional interactive demo → /demos/batchnorm
  layouts/
    BlogPostLayout.astro  # wraps every post (title, date, prose, comments)
    BaseLayout.astro      # site shell (nav, footer)
public/
  plots/<slug>/...        # images for a post (referenced as /plots/<slug>/file.png)
  <slug>/...              # optional static assets for a demo (e.g. an exported dashboard)
```

**Routing is file-based:** `src/pages/blog/foo.astro` is automatically served at
`/blog/foo`. There is **no** automatic post list — the `/blog` page is a
hand-written grid, so a new post is two steps: create the page **and** add a card
(see §4.3).

---

## 4. Create a new blog post

Say the new post's slug is `dropout`.

### 4.1 Add your images
Generate the plots in the experiment's own repo (§0), then **copy** them under
`public/plots/dropout/` here. Reference them in the post with
an **absolute** path from the site root, e.g. `/plots/dropout/curve.png`
(filenames with spaces are fine but URL-encode awkward characters; prefer
`snake-case.png`). Also add a ~landscape **card image** for the index
(e.g. `/plots/dropout/card.png`).

### 4.2 Create the post page
Create `src/pages/blog/dropout.astro`. Minimal starter template:

```astro
---
import BlogPostLayout from '../../layouts/BlogPostLayout.astro';

export const frontmatter = {
  title: 'Dropout, Visualized',
  pubDate: '2026-07-01',     // YYYY-MM-DD
};
---

<BlogPostLayout frontmatter={frontmatter}>
  <p class="mb-6 text-gray-700">One-sentence hook: what the reader will learn.</p>

  <div class="mb-6 p-4 rounded border bg-blue-50 text-blue-900">
    <b>TL;DR.</b> The headline result in two sentences.
  </div>

  <section class="mt-10">
    <h2 class="text-3xl md:text-4xl font-bold mb-3">A section</h2>
    <p class="text-gray-700 mb-3">Body text. Define jargon on first use.</p>

    <figure class="bg-white rounded border p-2 mb-2">
      <img src="/plots/dropout/curve.png" alt="Describe the figure" class="w-full rounded" />
    </figure>
  </section>

  <p class="mt-6 text-base md:text-lg text-gray-600">
    Caveat: say how many runs/seeds, what's measured, what's not.
  </p>
</BlogPostLayout>
```

The layout already renders the `title` as an `<h1>`, the `pubDate`, wraps your
content in a Tailwind Typography `prose` container, and adds the comment widget.
You only write the body.

#### Handy building blocks (copy from `batchnorm.astro`)
- **Callout boxes:** `class="p-4 rounded border bg-{blue|emerald|amber|rose|gray}-50 text-{...}-900"`.
- **Collapsible detail:** `<details><summary>…</summary>…</details>`.
- **Image carousel + click-to-enlarge lightbox (the coverflow used in the BatchNorm post):**
  ```html
  <div class="not-prose bn-carousel"
       data-images='["/plots/dropout/a.png","/plots/dropout/b.png"]'
       data-captions='["Caption A","Caption B"]'></div>
  ```
  ⚠️ The carousel's CSS (`<style is:global>`) and JS (`<script type="module">`)
  currently live **inside `batchnorm.astro`**. To reuse it in another post, either
  copy those two blocks into your new post, or (better) extract them into a shared
  component, e.g. `src/components/Carousel.astro`, and import it in both. The
  markup above only works on a page that includes that script.

### 4.3 Add the post to the `/blog` index
Open `src/pages/blog.astro` and add a card to the grid (copy an existing `<a>`
block and edit the href, image, title, and blurb):

```html
<a href="/blog/dropout" class="group block rounded-xl overflow-hidden border bg-white shadow-sm hover:shadow-md transition-shadow">
  <div class="h-32 sm:h-36 md:h-40 bg-gray-100 overflow-hidden flex items-center justify-center">
    <img src="/plots/dropout/card.png" alt="Dropout preview" class="max-h-full max-w-full object-contain group-hover:scale-[1.02] transition-transform duration-200" loading="lazy" />
  </div>
  <div class="p-4">
    <h2 class="text-lg font-semibold mb-1">Dropout, Visualized</h2>
    <p class="text-sm text-gray-600">One-line description for the card.</p>
  </div>
</a>
```

### 4.4 (Optional) Add an interactive demo
If the post links to a dashboard:
1. Drop the standalone asset in `public/dropout/` (e.g. an exported HTML dashboard).
2. Create `src/pages/demos/dropout.astro` that `<iframe>`s it (copy
   `src/pages/demos/batchnorm.astro`).
3. Link to `/demos/dropout` from the post.

### 4.5 Preview
```bash
npm run dev
```
Open `/blog`, click the new card, check the post and any carousels/demos.

---

## 5. Publish (push → auto-deploy)

**Where to push:** the `main` branch of
`eitamar-saraf/eitamar-saraf.github.io`. Every push to `main` triggers the
GitHub Actions workflow `.github/workflows/deploy.yml`, which runs
`npm ci && npm run build` on **Node 20** and deploys `./dist` to GitHub Pages
(usually live in ~1 minute).

### Direct to main (fastest)
```bash
git add src/pages/blog/dropout.astro src/pages/blog.astro public/plots/dropout
git commit -m "blog(dropout): add Dropout, Visualized"
git push origin main
```

### Via a branch + PR (safer, reviewable)
```bash
git checkout -b post/dropout
git add -A && git commit -m "blog(dropout): add post"
git push -u origin post/dropout
gh pr create --fill          # open a PR; merge to main to publish
```
Note: Pages only deploys from `main`, so a branch/PR won't appear on the live
site until it's merged.

### Watch the deploy
```bash
gh run list --limit 1        # status of the latest deploy
gh run watch                 # follow it live
```
Then verify the live page (append a dummy query like `?v=2` to dodge the CDN
cache): `https://eitamar-saraf.github.io/blog/dropout/?v=2`.

---

## 6. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Permission ... denied to eitamarSaraf` on push | Wrong account/protocol. Do the HTTPS + `gh` steps in §1.3 (`gh auth switch --user eitamar-saraf`, `gh auth setup-git`, set HTTPS remote). |
| `Node.js vX is not supported by Astro` | Node too old. `nvm install 20 && nvm use 20` (§1.1). |
| Images 404 on the live site | Asset must be under `public/` and referenced by an absolute path (`/plots/<slug>/file.png`), not a relative one. |
| New post not on `/blog` | You created the page but didn't add a card to `src/pages/blog.astro` (§4.3). |
| Carousel shows nothing / no buttons | The page is missing the carousel `<style>`/`<script>` blocks. Copy them from `batchnorm.astro` or use a shared component (§4.2). |
| Old content still showing after deploy | CDN cache. Hard-refresh or add `?v=N` to the URL. |
| Deploy succeeded but page didn't change | Check `gh run list` — confirm the run was for your commit SHA and concluded `success`. |

---

## 7. Quick reference

```bash
# setup
nvm use 20 && npm install
gh auth switch --user eitamar-saraf && gh auth setup-git

# write & preview
npm run dev                    # http://localhost:4321

# publish
git add -A
git commit -m "blog(<slug>): <summary>"
git push origin main           # → GitHub Actions → Pages (~1 min)
gh run watch                   # follow the deploy
```
