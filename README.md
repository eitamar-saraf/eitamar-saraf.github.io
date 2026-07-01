
# Eitamar Saraf Portfolio

This is a personal portfolio built with Astro, React, and Tailwind CSS. It features interactive machine learning demos, a blog, and a CV page.

## Structure
- **CV**: /cv (from `src/content/cv.json`)
- **Blog**: /blog (each post is a standalone `.astro` page in `src/pages/blog/`)
- **Demos**: /demos (interactive dashboards embedded via iframe)
- **Projects**: /projects (card grid and writeups)

## Getting Started
Requires **Node ≥ 18.20.8** (use `nvm install 20`).

1. Install dependencies: `npm install`
2. Start the dev server: `npm run dev` (http://localhost:4321)
3. Build for production: `npm run build`

## Deployment
Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds on Node 20
and deploys `./dist` to GitHub Pages.

## Writing a new blog post
See **[AUTHORING.md](./AUTHORING.md)** for a full guide: account/auth setup, how to
create a post and its index card, adding images and demos, the carousel/lightbox
component, and how & where to push.
